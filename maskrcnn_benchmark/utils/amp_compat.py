try:
    from maskrcnn_benchmark.utils.amp_compat import amp  # type: ignore
    HAS_APEX = True
except ImportError:
    HAS_APEX = False

    class _AmpCompat:
        @staticmethod
        def float_function(func):
            return func

        @staticmethod
        def initialize(model, optimizer=None, opt_level='O0', **kwargs):
            if optimizer is None:
                return model
            return model, optimizer

        @staticmethod
        def scale_loss(loss, optimizer):
            class _ScaleLossCtx:
                def __init__(self, loss):
                    self.loss = loss
                def __enter__(self):
                    return self.loss
                def __exit__(self, exc_type, exc, tb):
                    return False
            return _ScaleLossCtx(loss)

    amp = _AmpCompat()
