from ..utils.normalizer import LinearNormalizer


class FastWAMProcessor:
    def __init__(self, shape_meta, action_output_dim, proprio_output_dim):
        self.shape_meta = shape_meta
        self.normalizer = None
        for group, width in (("action", action_output_dim), ("state", proprio_output_dim)):
            fields = shape_meta[group]
            if len(fields) != 1 or fields[0]["shape"] != width or fields[0]["raw_shape"] != width:
                raise ValueError(f"Expected one absolute-joint {group} field of width {width}")

    def set_normalizer_from_stats(self, dataset_stats):
        self.normalizer = LinearNormalizer(
            shape_meta=self.shape_meta,
            use_stepwise_action_norm=False,
            default_mode="z-score",
            exception_mode=None,
            stats=dataset_stats,
        )

    def action_state_transform(self, batch):
        for group in ("state", "action"):
            if group == "action" and group not in batch:
                continue
            field = self.shape_meta[group][0]
            if batch[group][field["key"]].shape[-1] != field["shape"]:
                raise ValueError(f"Unexpected {group} dimension")
        return batch
