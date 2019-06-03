from delira import get_backends

if "MX" in get_backends():
    import mxnet
    from .abstract_network import AbstractMXNetwork

    def _apply_scatter(inputs: mxnet.nd.NDArray, target_devices: list,
                       dim: int = 0):
        """
        Scatters inputs to target devices; Slicing will be done against a given
        dimension

        Parameters
        ----------
        inputs : :class:`mxnet.nd.NDArray`
            the input variable to scatter
        target_devices : list
            the target devices to scatter to
        dim : int
            the dimension to use for slicing

        Returns
        -------
        list
            list of variable slices on correct devices
        """

        def _slice_inputs(input_var, dim, num_dims, start, end, target_device):
            """
            Slices the input variable along a given dimension from start to end
            and pushes it to correct device

            Parameters
            ----------
            input_var : :class:`mxnet.nd.NDArray`
                the variable to slice
            dim : int
                the dimension to slice along
            num_dims : int
                the dimensionality of ``input_var``
            start : int
                the start value for slicing (included)
            end : int
                the end value for slicing (excluded)
            target_device: :class:`mxnet.context.Context`
                the device to push to

            Returns
            -------
            :class:`mxnet.nd.NDArray`
                the slice of the variable
            """
            slc = [slice(None)] * num_dims
            slc[dim] = slice(start, end)
            sliced_var = input_var[slc]
            sliced_var = sliced_var.copyto(target_device)
            output_shape = list(input_var.shape)
            output_shape[dim] = -1
            return sliced_var.reshape(output_shape)

        # create empty sliced input list
        scattered_inputs = []

        # calculate constant only once
        num_devices = len(target_devices)
        samples_per_device = inputs.shape[dim] // num_devices
        num_dims = len(inputs.shape)

        # iterate over number of devices and slice accordingly
        # (exclude last device)
        # iterating until the minimum of num_devices and inputs.shape[dim] -1
        # ensures that if the batchsize is too small to be scattered across all
        # devices, we will only scatter across as many devices as possible
        for i in range(min(num_devices, inputs.shape[dim])-1):
            start, end = i * samples_per_device, i + 1 * samples_per_device
            scattered_inputs.append(_slice_inputs(inputs, dim,
                                                  num_dims, start, end,
                                                  target_devices[i]))

        # all remaining samples (not yet sliced) are appended now
        # (all samples used; will be pushed to last device later)
        scattered_inputs.append(_slice_inputs(
            inputs, dim, len(inputs.shape,),
            (num_devices - 1) * samples_per_device,
            inputs.shape[dim], target_devices[-1]))

        return scattered_inputs

    def _apply_gather(target_device, dim, *outputs):
        for _output in outputs:
            _output.to_device(target_device)

        return mxnet.nd.concatenate(outputs, dim)

    def _scatter(inputs, target_devices: list, dim):
        """
        Scatters all inputs across given target_devices

        Parameters
        ----------
        inputs : Any
        target_devices : list
            list of devices to scatter to
        dim : int
            dimension to use for slicing
        Returns
        -------
        list
            list of scattered inputs
        """

        def _scatter_map(inputs):
            """
            Scatters all inputs across given target_devices

            Parameters
            ----------
            inputs : Any
            Returns
            -------
            list
                list of scattered inputs
            """

            # directly apply the scattering on variable
            if isinstance(inputs, mxnet.nd.NDArray):
                return _apply_scatter(inputs, target_devices, dim)

            # map _scatter_map recursively to all samples in tuple
            if isinstance(inputs, tuple) and len(inputs) > 0:
                return list(zip(*map(_scatter_map, inputs)))

            # map _scatter_map recursively to all samples in list
            if isinstance(inputs, list) and len(inputs) > 0:
                return list(map(list, zip(*map(_scatter_map,
                                               inputs))))

            # map _scatter_map recursively to all samples in dict
            if isinstance(inputs, dict) and len(inputs) > 0:
                return list(map(type(inputs), zip(*map(_scatter_map,
                                                       inputs.items()))))

            # try to convert inputs to chainer variable first and afterwards apply
            # _scatter_map again
            try:
                return _scatter_map(mxnet.nd.array(inputs))
            except TypeError:
                return [inputs for targets in target_devices]

        # After scatter_map is called, a scatter_map cell will exist. This cell
        # has a reference to the actual function scatter_map, which has references
        # to a closure that has a reference to the scatter_map cell (because the
        # fn is recursive). To avoid this reference cycle, we set the function to
        # None, clearing the cell
        try:
            return _scatter_map(inputs)
        finally:
            _scatter_map = None

    def _gather(outputs, target_device, dim=0):
        r"""
        Gathers tensors from different GPUs on a specified device
          (-1 means the CPU).
        """

        def gather_map(outputs):
            out = outputs[0]
            if isinstance(out, mxnet.nd.NDArray):
                return _apply_gather(target_device, dim, *outputs)
            if out is None:
                return None
            if isinstance(out, dict):
                if not all((len(out) == len(d) for d in outputs)):
                    raise ValueError('All dicts must have the same number of keys')
                return type(out)(((k, gather_map([d[k] for d in outputs]))
                                  for k in out))
            return type(out)(map(gather_map, zip(*outputs)))

        # Recursive function calls like this create reference cycles.
        # Setting the function to None clears the refcycle.
        try:
            return gather_map(outputs)
        finally:
            gather_map = None


    class DataParallel(mxnet.gluon.Block):
        """
        A Wrapper around a ``mxnet.gluon.Block`` instance to implement parallel
        training by splitting the batches
        """
        def __init__(self, module: AbstractMXNetwork, devices: list,
                     output_device=None, batch_dim=0):
            """

            Parameters
            ----------
            module : :class:`mxnet.gluon.Block`
                the module to wrap (will be replicated on all devices)
            devices : list
                a list containing the devices to use (either as strings or as
                chainer.backend.Device). The first device will be used as output
                device. Make sure, your labels are also on this device for loss
                calculation!
            output_device : :class:`mxnet.context.Context`
                the device to accumulate outputs to. If None: uses the first
                item of ``devices``; default: None
            batch_dim : int
                the index of the batchdimension (usually 0, but can become
                e.g. 1 in NLP tasks)
            """
            super().__init__()

            self.modules = module
            self.modules.initialize(ctx=devices, force_reinit=True)

            self.devices = devices

            if output_device is None:
                output_device = devices[0]
            self._output_device = output_device
            assert self._output_device in self.devices
            self._output_device_idx = self.devices.index(self._output_device)
            self.dim = batch_dim

        def forward(self, *args, **kwargs):
            """
            Scatters the inputs (both positional and keyword arguments) across
            all devices, feeds them through model replicas and re-builds
            batches on output device

            Parameters
            ----------
            *args :
                positional arguments of arbitrary number and type
            **kwargs :
                keyword arguments of arbitrary number and type
            Returns
            -------
            """
            scattered_args, scattered_kwargs = self._scatter(args, kwargs,
                                                             self.devices,
                                                             self.dim)

            predictions = []
            for _args, _kwargs in zip(scattered_args, scattered_kwargs):
                predictions.append(self.modules(*_args, **_kwargs))

            predictions = self._gather(predictions, self.dim,
                                       self._output_device)

            return predictions

        @staticmethod
        def _scatter(inputs, kwargs, target_devices: list, dim=0):
            """
            Scatters all inputs (args and kwargs) to target devices and splits
            along given dimension

            Parameters
            ----------
            inputs : list or tuple
                positional arguments
            kwargs : dict
                keyword arguments
            target_devices : list
                list of target devices (each item must be of type
                :class:`mxnet.context.Context`)
            dim : int
                the dimension, which should be used for splitting the batch

            Returns
            -------
            tuple
                scattered positional arguments
            tuple
                scattered keyword arguments
            """

            # scatter inputs if given
            inputs = _scatter(inputs, target_devices, dim) if inputs else []
            # scatter kwargs if given
            kwargs = _scatter(kwargs, target_devices, dim) if kwargs else []

            # extend lengths by empty tuples if necessary
            if len(inputs) < len(kwargs):
                inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
            elif len(kwargs) < len(inputs):
                kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

            inputs = tuple(inputs)
            kwargs = tuple(kwargs)

            return inputs, kwargs

        @staticmethod
        def _gather(predictions, dim, target_device):
            """
            Re-Builds batches on the target device

            Parameters
            ----------
            predictions : list
                list containing the predictions from all replicated models
            dim : int
                dimension to use for concatenating single predictions
            target_device : :class:`mxnet.context.Context`
                the device, the re-built batch should lie on

            Returns
            -------
            Any
                the rebuild batch (lying on ``target_device``)
            """

            return _gather(predictions, target_device, dim)

