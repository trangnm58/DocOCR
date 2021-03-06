from __future__ import division, print_function, unicode_literals
import cv2
import numpy as np
import scipy.optimize


PAGE_MARGIN_X = 0  # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 0  # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0  # how much to zoom output relative to *original* image
OUTPUT_DPI = 300  # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 8  # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55  # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 15  # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2  # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5  # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 20  # max reduced px thickness of detected text contour

EDGE_MAX_OVERLAP = 1.0  # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0  # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5  # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)  # index of rvec in params vector
TVEC_IDX = slice(3, 6)  # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30  # minimum reduced px width for span
SPAN_PX_PER_STEP = 50  # reduced px spacing for sampling along spans
FOCAL_LENGTH = 2.0  # normalized focal length of camera

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)


class SimplePreProcess():
    NORM_PX = 2000  # the shorter side must have this resolution

    def resize(self, image):
        height, width = image.shape[:2]
        if height < width:
            target_width = round((SimplePreProcess.NORM_PX / height) * width)
            img_res = cv2.resize(src=image, dsize=(target_width, SimplePreProcess.NORM_PX), interpolation=cv2.INTER_CUBIC)
        else:
            target_height = round((SimplePreProcess.NORM_PX / width) * height)
            img_res = cv2.resize(src=image, dsize=(SimplePreProcess.NORM_PX, target_height), interpolation=cv2.INTER_CUBIC)

        return img_res

    def binarization(self, image):
        bi_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 25)
        return bi_img

    def deskew(self, image):
        # invert color: text is white, background is black
        gray = cv2.bitwise_not(image)

        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return rotated

    def process(self, image):
        # convert to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # resize and redude size
        img_res = self.resize(gray)
        # binarization
        img_bi = self.binarization(img_res)
        # deskew
        result = self.deskew(img_bi)
        
        return result


class ContourInfo():
    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = self._blob_mean_and_tangent(contour)
        
        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]  # project all points into Ox
        
        lxmin = np.min(clx)
        lxmax = np.max(clx)
        
        self.local_xrng = (lxmin, lxmax)

        self.point0 = self.center + self.tangent * lxmin  # the first point
        self.point1 = self.center + self.tangent * lxmax  # the last point

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten() - self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return self._interval_measure_overlap(self.local_xrng, (xmin, xmax))

    def _interval_measure_overlap(self, int_a, int_b):
        return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])  # overlap of 2 contours

    def _blob_mean_and_tangent(self, contour):
        moments = cv2.moments(contour)

        area = moments['m00']

        mean_x = moments['m10'] / area
        mean_y = moments['m01'] / area

        moments_matrix = np.array([
            [moments['mu20'], moments['mu11']],
            [moments['mu11'], moments['mu02']]
        ]) / area

        _, svd_u, _ = cv2.SVDecomp(moments_matrix)

        center = np.array([mean_x, mean_y])
        tangent = svd_u[:, 0].flatten().copy()

        return center, tangent


class AdvancedPreProcess():
    def resize_to_screen(self, src, norm_w=700, norm_h=1280, copy=False):  # modified
        height, width = src.shape[:2]

        scl_x = width / norm_w
        scl_y = height / norm_h

        scl = max(scl_x, scl_y)

        if copy:
            img = src.copy()
        else:
            inv_scl = 1.0 / scl
            img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)

        return img

    def get_page_extents(self, small):
        height, width = small.shape[:2]
    
        xmin = PAGE_MARGIN_X
        ymin = PAGE_MARGIN_Y
        xmax = width - PAGE_MARGIN_X
        ymax = height - PAGE_MARGIN_Y

        page = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)
    
        outline = np.array([
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin]])
    
        return page, outline
    
    def box(self, width, height):
        return np.ones((height, width), dtype=np.uint8)
    
    def get_mask(self, small, pagemask, masktype):
        sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        if masktype == 'text':
            mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_WINSZ, 25)
            mask = cv2.dilate(mask, self.box(9, 1))
            mask = cv2.erode(mask, self.box(1, 3))
        else:
            mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_WINSZ, 7)
            mask = cv2.erode(mask, self.box(3, 1), iterations=3)
            mask = cv2.dilate(mask, self.box(8, 2))

        return np.minimum(mask, pagemask)
    
    def make_tight_mask(self, contour, xmin, ymin, width, height):
        tight_mask = np.zeros((height, width), dtype=np.uint8)
        # move to O(0,0) because contour are points from big image
        tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    
        cv2.drawContours(tight_mask, [tight_contour], 0, (1, 1, 1), -1)
    
        return tight_mask
    
    def get_contours(self, small, pagemask, masktype):
        mask = self.get_mask(small, pagemask, masktype)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours_out = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            xmin, ymin, width, height = rect
            
            if (width < TEXT_MIN_WIDTH
                or height < TEXT_MIN_HEIGHT
                or width < TEXT_MIN_ASPECT * height):
                continue
    
            tight_mask = self.make_tight_mask(contour, xmin, ymin, width, height)
            
            if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
                continue

            contours_out.append(ContourInfo(contour, rect, tight_mask))

        return contours_out
    
    def angle_dist(self, angle_b, angle_a):
        diff = angle_b - angle_a
        while diff > np.pi:
            diff -= 2*np.pi
        while diff < -np.pi:
            diff += 2*np.pi
        return np.abs(diff)
    
    def generate_candidate_edge(self, cinfo_a, cinfo_b):
        # we want a left of b (so a's successor will be b and b's
        # predecessor will be a) make sure right endpoint of b is to the
        # right of left endpoint of a.
        if cinfo_a.point0[0] > cinfo_b.point1[0]:
            cinfo_a, cinfo_b = cinfo_b, cinfo_a
    
        x_overlap_a = cinfo_a.local_overlap(cinfo_b)
        x_overlap_b = cinfo_b.local_overlap(cinfo_a)
    
        overall_tangent = cinfo_b.center - cinfo_a.center
        overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])
    
        delta_angle = max(self.angle_dist(cinfo_a.angle, overall_angle),
                          self.angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi  # in degrees
    
        # we want the largest overlap in x to be small
        x_overlap = max(x_overlap_a, x_overlap_b)
    
        dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)
    
        if (dist > EDGE_MAX_LENGTH
            or x_overlap > EDGE_MAX_OVERLAP
            or delta_angle > EDGE_MAX_ANGLE):
            return None
        else:
            score = dist + delta_angle * EDGE_ANGLE_COST
            return (score, cinfo_a, cinfo_b)
    
    def assemble_spans(self, cinfo_list):
        # sort list by ymin of contours
        cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])
    
        # generate all candidate edges
        candidate_edges = []
        for i, cinfo_i in enumerate(cinfo_list):
            for j in range(i):
                # note e is of the form (score, left_cinfo, right_cinfo)
                edge = self.generate_candidate_edge(cinfo_i, cinfo_list[j])
                if edge is not None:
                    candidate_edges.append(edge)
    
        # sort candidate edges by score (lower is better)
        candidate_edges.sort()
    
        # for each candidate edge
        for _, cinfo_a, cinfo_b in candidate_edges:
            # if left and right are unassigned, join them
            if cinfo_a.succ is None and cinfo_b.pred is None:
                cinfo_a.succ = cinfo_b
                cinfo_b.pred = cinfo_a
    
        # generate list of spans as output
        spans = []
        # until we have removed everything from the list
        while cinfo_list:
            # get the first on the list
            cinfo = cinfo_list[0]
            # keep following predecessors until none exists
            while cinfo.pred:
                cinfo = cinfo.pred
            # start a new span
            cur_span = []
            width = 0.0
            # follow successors til end of span
            while cinfo:
                # remove from list (sadly making this loop *also* O(n^2)
                cinfo_list.remove(cinfo)
                # add to span
                cur_span.append(cinfo)
                width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
                # set successor
                cinfo = cinfo.succ
    
            # add if long enough
            if width > SPAN_MIN_WIDTH:
                spans.append(cur_span)
    
        return spans  # list of list of contour objects
    
    def pix2norm(self, shape, pts):
        """
        x = (x - width/2) * 2/max(width, height)
        y = (y - height/2) * 2/max(width, height)
        Range of norms:
        -w/max(w,h) < norm_x < w/max(w,h)
        -h/max(w,h> < norm_y < h/max(w,h)
        """
        height, width = shape[:2]
        scl = 2.0/(max(height, width))
        offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
        return (pts - offset) * scl
    
    def norm2pix(self, shape, pts, as_integer):
        height, width = shape[:2]
        scl = max(height, width)*0.5
        offset = np.array([0.5*width, 0.5*height], dtype=pts.dtype).reshape((-1, 1, 2))
        rval = pts * scl + offset
        if as_integer:
            return (rval + 0.5).astype(int)
        else:
            return rval
    
    def sample_spans(self, shape, spans):
        span_points = []
        for span in spans:
            contour_points = []
            for cinfo in span:
                yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
                totals = (yvals * cinfo.mask).sum(axis=0)
                means = totals / cinfo.mask.sum(axis=0)  # mean y (row) of contour
                
                xmin, ymin = cinfo.rect[:2]
    
                step = SPAN_PX_PER_STEP  # one point each step
                start = int(((len(means)-1) % step) / 2)
    
                contour_points += [(x+xmin, means[x]+ymin)
                                   for x in range(start, len(means), step)]

            contour_points = np.array(contour_points, dtype=np.float32).reshape((-1, 1, 2))
            contour_points = self.pix2norm(shape, contour_points)
            span_points.append(contour_points)

        return span_points
    
    def keypoints_from_samples(self, pagemask, page_outline, span_points):
        all_evecs = np.array([[0.0, 0.0]])
        all_weights = 0
        for points in span_points:
            _, evec = cv2.PCACompute(points.reshape((-1, 2)), None, maxComponents=1)

            weight = np.linalg.norm(points[-1] - points[0])  # Frobenius norm

            all_evecs += evec * weight
            all_weights += weight
    
        evec = all_evecs / all_weights

        x_dir = evec.flatten()

        if x_dir[0] < 0:
            x_dir = -x_dir
    
        y_dir = np.array([-x_dir[1], x_dir[0]])
        
        pagecoords = cv2.convexHull(page_outline)
        pagecoords = self.pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
        pagecoords = pagecoords.reshape((-1, 2))
        
        px_coords = np.dot(pagecoords, x_dir)
        py_coords = np.dot(pagecoords, y_dir)

        px0 = px_coords.min()
        px1 = px_coords.max()
    
        py0 = py_coords.min()
        py1 = py_coords.max()
    
        p00 = px0 * x_dir + py0 * y_dir
        p10 = px1 * x_dir + py0 * y_dir
        p11 = px1 * x_dir + py1 * y_dir
        p01 = px0 * x_dir + py1 * y_dir
    
        corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

        ycoords = []
        xcoords = []  # list of (list of point's x coordinate of each line)
        for points in span_points:
            pts = points.reshape((-1, 2))
            px_coords = np.dot(pts, x_dir)
            py_coords = np.dot(pts, y_dir)
            ycoords.append(py_coords.mean() - py0)
            xcoords.append(px_coords - px0)
    
        return corners, np.array(ycoords), xcoords
    
    def get_default_params(self, corners, ycoords, xcoords):
        # page width and height
        page_width = np.linalg.norm(corners[1] - corners[0])
        page_height = np.linalg.norm(corners[-1] - corners[0])
        rough_dims = (page_width, page_height)

        # our initial guess for the cubic has no slope
        cubic_slopes = [0.0, 0.0]
    
        # object points of flat page in 3D coordinates
        corners_object3d = np.array([
            [0, 0, 0],
            [page_width, 0, 0],
            [page_width, page_height, 0],
            [0, page_height, 0]])
    
        # estimate rotation and translation from four 2D-to-3D point
        # correspondences
        _, rvec, tvec = cv2.solvePnP(corners_object3d, corners, K, np.zeros(5))  # rotation and translation vectors
    
        span_counts = [len(xc) for xc in xcoords]

        params = np.hstack((np.array(rvec).flatten(), np.array(tvec).flatten(), np.array(cubic_slopes).flatten(), ycoords.flatten()) + tuple(xcoords))
        return rough_dims, span_counts, params
    
    def make_keypoint_index(self, span_counts):
        nspans = len(span_counts)
        npts = sum(span_counts)
        keypoint_index = np.zeros((npts+1, 2), dtype=int)
        start = 1
    
        for i, count in enumerate(span_counts):
            end = start + count
            keypoint_index[start:start+end, 1] = 8+i
            start = end
    
        keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans
        return keypoint_index
    
    def project_xy(self, xy_coords, pvec):
        # get cubic polynomial coefficients given
        #
        #  f(0) = 0, f'(0) = alpha
        #  f(1) = 0, f'(1) = beta
        alpha, beta = tuple(pvec[CUBIC_IDX])
        poly = np.array([alpha + beta, -2*alpha - beta, alpha, 0])  # (alpha + beta)*x**3 - (2*alpha + beta)*x**2 + alpha*x + 0

        xy_coords = xy_coords.reshape((-1, 2))
        z_coords = np.polyval(poly, xy_coords[:, 0])
    
        objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))  # 3D points

        image_points, _ = cv2.projectPoints(objpoints, pvec[RVEC_IDX], pvec[TVEC_IDX], K, np.zeros(5))  # 2D points
    
        return image_points
    
    def project_keypoints(self, pvec, keypoint_index):
        xy_coords = pvec[keypoint_index]
        xy_coords[0, :] = 0
    
        return self.project_xy(xy_coords, pvec)
    
    def optimize_params(self, small, dstpoints, span_counts, params):
        keypoint_index = self.make_keypoint_index(span_counts)

        def objective(pvec):
            ppts = self.project_keypoints(pvec, keypoint_index)
            return np.sum((dstpoints - ppts)**2)

        res = scipy.optimize.minimize(objective, params, method='Powell')
        params = res.x
    
        return params
    
    def get_page_dims(self, corners, rough_dims, params):  # modified
        dst_tr = corners[1].flatten()
        dst_br = corners[2].flatten()
        dims = np.array(rough_dims)
        def objective(dims):
            proj_tr = self.project_xy(np.array([dims[0], 0]), params)
            proj_br = self.project_xy(dims, params)  # 'br' - bottom right
            return np.sum((dst_br - proj_br.flatten())**2 + (dst_tr - proj_tr.flatten())**2)
    
        res = scipy.optimize.minimize(objective, dims, method='Powell')
        dims = res.x
        return dims
    
    def round_nearest_multiple(self, i, factor):
        i = int(i)
        rem = i % factor
        if not rem:
            return i
        else:
            return i + factor - rem
    
    def remap_image(self, img, small, page_dims, params):
        height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
        height = self.round_nearest_multiple(height, REMAP_DECIMATE)
    
        width = self.round_nearest_multiple(height * page_dims[0] / page_dims[1], REMAP_DECIMATE)
    
        height_small = height / REMAP_DECIMATE
        width_small = width / REMAP_DECIMATE
        
        page_x_range = np.linspace(0, page_dims[0], num=width_small)
        page_y_range = np.linspace(0, page_dims[1], num=height_small)

        page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
    
        page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)), page_y_coords.flatten().reshape((-1, 1))))
    
        page_xy_coords = page_xy_coords.astype(np.float32)
    
        image_points = self.project_xy(page_xy_coords, params)
        image_points = self.norm2pix(img.shape, image_points, False)

        image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
        image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)
    
        image_x_coords = cv2.resize(image_x_coords, (width, height), interpolation=cv2.INTER_CUBIC)
    
        image_y_coords = cv2.resize(image_y_coords, (width, height), interpolation=cv2.INTER_CUBIC)
    
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        remapped = cv2.remap(img_gray, image_x_coords, image_y_coords, cv2.INTER_CUBIC, None, cv2.BORDER_REPLICATE)

        thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADAPTIVE_WINSZ, 25)
    
        return thresh
    
    def process(self, img):
        small = self.resize_to_screen(img)
        
        pagemask, page_outline = self.get_page_extents(small)
        
        cinfo_list = self.get_contours(small, pagemask, 'text')
        
        spans = self.assemble_spans(cinfo_list)  # list of list of contour objects
        
        if len(spans) < 3:
            cinfo_list = self.get_contours(small, pagemask, 'line')
            spans2 = self.assemble_spans(cinfo_list)
            if len(spans2) > len(spans):
                spans = spans2
    
        if len(spans) < 1:
            return None
        
        span_points = self.sample_spans(small.shape, spans)  # list of points

        corners, ycoords, xcoords = self.keypoints_from_samples(pagemask, page_outline, span_points)
      
        rough_dims, span_counts, params = self.get_default_params(corners, ycoords, xcoords)
      
        dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))  # destination points
      
        params = self.optimize_params(small, dstpoints, span_counts, params)
      
        page_dims = self.get_page_dims(corners, rough_dims, params)
        
        outfile = self.remap_image(img, small, page_dims, params)

        return outfile

if __name__ == '__main__':
    p = AdvancedPreProcess()
    img = cv2.imread('08_right.png')
    out = p.process(img)
    res = cv2.resize(out, dsize=(0,0), fx=1/6, fy=1/6, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    