import torch

TEMPLATE_POINTS_NUM = 17


class TemplateGenerator(object):

    def __init__(self, base_size, scales, templates, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.templates = torch.Tensor(templates)
        self.templates = self.templates[:, :, 0:2]
        self.templates = self.templates.reshape((self.templates.shape[0], -1))
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors, self.base_anchors_scales = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        ts = []
        ss = []
        if self.scale_major:
            for n_c in self.templates:
                for n_s in self.scales:
                    ts.append((w * n_c * n_s).view(1, -1))
                    ss.append(w * n_s)
        else:
            for n_s in self.scales:
                for n_c in self.templates:
                    ts.append((w * n_c * n_s).view(1, -1))
                    ss.append(w * n_s)
        ts = torch.cat(ts)
        ss = torch.stack(ss)
        ss = ss.view(-1, 1)

        # move to the center
        ts = ts.reshape(ts.shape[0], -1, 2)
        ts[:, :, 0] = ts[:, :, 0] + x_ctr
        ts[:, :, 1] = ts[:, :, 1] + y_ctr
        base_anchors = ts.reshape(ts.shape[0], -1).round()

        return base_anchors, ss

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)
        base_anchors_scales = self.base_anchors_scales.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

        shifts = []
        for n in range(0, int(base_anchors.shape[-1] / 2)):
            shifts.append(shift_xx)
            shifts.append(shift_yy)
        shifts = torch.stack(shifts, dim=-1)

        shifts = shifts.type_as(base_anchors)

        shifts_for_scale = shifts[:, 0:1].clone().detach()
        shifts_for_scale = shifts_for_scale * 0

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 2 * TEMPLATE_POINTS_NUM)

        zero_anchors = base_anchors.new_zeros(base_anchors.shape)
        all_zero_anchors = zero_anchors[None, :, :] + shifts[:, None, :]
        all_zero_anchors = all_zero_anchors.view(-1, 2 * TEMPLATE_POINTS_NUM)

        all_anchors_scales = base_anchors_scales[None, :, :] + shifts_for_scale[:, None, :]
        all_anchors_scales = all_anchors_scales.view(-1, 1)

        return all_anchors, all_zero_anchors, all_anchors_scales

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
