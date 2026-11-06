"""Label upstream expert-distribution diagnostics; never enable for throughput."""
import _bootstrap
def enable():
    from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal
    cls = _ExpertDistributionRecorderReal
    if getattr(cls, "_profile_spec_labels", False):
        return
    original_start, original_end = cls._on_forward_pass_start, cls._on_forward_pass_end

    def start(self, batch):
        self._profile_spec_metadata = {
            "forward_mode": batch.forward_mode.name,
            "input_token_rows": int(batch.input_ids.numel()),
            "request_rows": int(batch.req_pool_indices.numel()),
        }
        return original_start(self, batch)

    def end(self, forward_pass_id, outputs):
        records = getattr(self._accumulator, "_records", [])
        before = len(records)
        result = original_end(self, forward_pass_id, outputs)
        for record in records[before:]:
            record["profile_spec_metadata"] = dict(self._profile_spec_metadata)
        return result

    cls._on_forward_pass_start, cls._on_forward_pass_end = start, end
    cls._profile_spec_labels = True
