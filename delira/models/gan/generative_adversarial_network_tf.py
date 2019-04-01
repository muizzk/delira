import logging
import typing
import numpy as np
from delira import get_backends

logger = logging.getLogger(__name__)


if "TF" in get_backends():
    import tensorflow as tf
    from delira.models.abstract_network import AbstractTfNetwork
    tf.keras.backend.set_image_data_format('channels_first')
    class GenerativeAdversarialNetworkBaseTf(AbstractTfNetwork):
        """Implementation of Vanilla DC-GAN to create 64x64 pixel images

        Notes
        -----
        The fully connected part in the discriminator has been replaced with an
        equivalent convolutional part

        References
        ----------
        https://arxiv.org/abs/1511.06434

        See Also
        --------
        :class:`AbstractTfNetwork`

        """

        def __init__(self, n_channels: int, noise_length: int, **kwargs):
            """

            Constructs graph containing model definition and forward pass behavior

            Parameters
            ----------
            n_channels : int
                number of image channels for generated images and input images
            noise_length : int
                length of noise vector
            **kwargs :
                additional keyword arguments

            """
            # register params by passing them as kwargs to parent class __init__
            super().__init__(n_channels=n_channels,
                            noise_length=noise_length,
                            **kwargs)

            # build on CPU for tf.keras.utils.multi_gpu_model() in tf_trainer.py
            # with tf.device('/cpu:0'):

            gen, discr = self._build_models(n_channels, **kwargs)

            self.noise_length = noise_length
            self.gen = gen
            self.discr = discr

            real_images = tf.placeholder(shape=[None, n_channels, 64, 64], dtype=tf.float32)
            fake_images = self.gen(tf.random.normal(shape=(tf.shape(real_images)[0], self.noise_length, 1, 1)))

            discr_real = self.discr(real_images)
            discr_fake = self.discr(fake_images)


            self.inputs = [real_images]
            self.outputs_train = [fake_images, discr_real, discr_fake]
            self.outputs_eval = [fake_images, discr_real, discr_fake]

            for key, value in kwargs.items():
                setattr(self, key, value)

        def _add_losses(self, losses: dict):
            """
            Adds losses to model that are to be used by optimizers or during evaluation

            Parameters
            ----------
            losses : dict
                dictionary containing all losses. Individual losses are averaged for discr_real, discr_fake and gen
            """
            if self._losses is not None and len(losses) != 0:
                logging.warning('Change of losses is not yet supported')
                raise NotImplementedError()
            elif self._losses is not None and len(losses) == 0:
                pass
            else:
                self._losses = {}

                total_loss_discr_fake = []
                total_loss_discr_real = []
                total_loss_gen = []


                for name, _loss in losses.items():
                    loss_val = _loss(tf.ones_like(self.outputs_train[1]), self.outputs_train[1])
                    self._losses[name + 'discr_real'] = loss_val
                    total_loss_discr_real.append(loss_val)

                for name, _loss in losses.items():
                    loss_val = _loss(tf.zeros_like(self.outputs_train[2]), self.outputs_train[2])
                    self._losses[name + 'discr_fake'] = loss_val
                    total_loss_discr_fake.append(loss_val)

                for name, _loss in losses.items():
                    loss_val = _loss(tf.ones_like(self.outputs_train[2]), self.outputs_train[2])
                    self._losses[name + 'gen'] = loss_val
                    total_loss_gen.append(loss_val)

                total_loss_discr = tf.reduce_mean([*total_loss_discr_real, *total_loss_discr_fake], axis=0)
                self._losses['total_discr'] = total_loss_discr

                total_loss_gen = tf.reduce_mean(total_loss_gen, axis=0)
                self._losses['total_gen'] = total_loss_gen

                self.outputs_train.append(self._losses)
                self.outputs_eval.append(self._losses)

        def _add_optims(self, optims: dict):
            """
            Adds optims to model that are to be used by optimizers or during training

            Parameters
            ----------
            optim: dict
                dictionary containing all optimizers, optimizers should be of Type[tf.train.Optimizer]
            """
            if self._optims is not None and len(optims) != 0:
                logging.warning('Change of optims is not yet supported')
                pass
                #raise NotImplementedError()
            elif self._optims is not None and len(optims) == 0:
                pass
            else:
                self._optims = optims

                optim_gen = self._optims['gen']
                grads_gen = optim_gen.compute_gradients(self._losses['total_gen'], self.gen.trainable_variables)
                step_gen = optim_gen.apply_gradients(grads_gen)

                optim_discr = self._optims['discr']
                grads_discr = optim_discr.compute_gradients(self._losses['total_discr'], self.discr.trainable_variables)
                step_discr = optim_discr.apply_gradients(grads_discr)

                steps = tf.group([step_gen, step_discr])

                self.outputs_train.append(steps)

        @staticmethod
        def _build_models(n_channels: int, **kwargs):
            """
            builds generator and discriminators

            Parameters
            ----------
            n_channels : int
                number of channels for fake and real images
            **kwargs :
                additional keyword arguments
            Returns
            -------
            tf.keras.Sequential
                created gen
            tf.keras.Sequential
                created discr
            """
            gen = tf.keras.models.Sequential(
                [
                     tf.keras.layers.Conv2DTranspose(64 * 8, 4, 1, use_bias=False),
                     tf.keras.layers.BatchNormalization(axis=1),
                     tf.keras.layers.ReLU(),
                     # state size. (64*8) x 4 x 4,
                     tf.keras.layers.Conv2DTranspose(64 * 4, 4, 2, padding='same', use_bias=False),
                     tf.keras.layers.BatchNormalization(axis=1),
                     tf.keras.layers.ReLU(),
                     # state size.
                     tf.keras.layers.Conv2DTranspose(64 * 2, 4, 2, padding='same', use_bias=False),
                     tf.keras.layers.BatchNormalization(axis=1),
                     tf.keras.layers.ReLU(),
                     # state size.
                     tf.keras.layers.Conv2DTranspose(64, 4, 2, padding='same', use_bias=False),
                     tf.keras.layers.BatchNormalization(axis=1),
                     tf.keras.layers.ReLU(),
                     # state size.
                     tf.keras.layers.Conv2DTranspose(n_channels, 4, 2, padding='same', use_bias=False),
                     tf.keras.layers.Activation('tanh')
                ]
            )

            discr = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Conv2D(64, 4, 2, padding='same', use_bias=False),
                    tf.keras.layers.LeakyReLU(0.2),

                    tf.keras.layers.Conv2D(64*2, 4, 2, padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(axis=1),
                    tf.keras.layers.LeakyReLU(0.2),

                    tf.keras.layers.Conv2D(64*4, 4, 2, padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(axis=1),
                    tf.keras.layers.LeakyReLU(0.2),

                    tf.keras.layers.Conv2D(64 * 8, 4, 2, padding='same', use_bias=False),
                    tf.keras.layers.BatchNormalization(axis=1),
                    tf.keras.layers.LeakyReLU(0.2),

                    tf.keras.layers.Conv2D(1, 4, 1, use_bias=False),
                    tf.keras.layers.Activation('sigmoid')
                ]
            )

            return gen, discr

        @staticmethod
        def closure(model: typing.Type[AbstractTfNetwork], data_dict: dict,
                    metrics={}, fold=0, **kwargs):
            """
                    closure method to do a single prediction.
                    This is followed by backpropagation or not based state of
                    on model.train

                    Parameters
                    ----------
                    model: AbstractTfNetwork
                        AbstractTfNetwork or its child-clases
                    data_dict : dict
                        dictionary containing the data
                    metrics : dict
                        dict holding the metrics to calculate
                    fold : int
                        Current Fold in Crossvalidation (default: 0)
                    **kwargs:
                        additional keyword arguments

                    Returns
                    -------
                    dict
                        Metric values (with same keys as input dict metrics)
                    dict
                        Loss values (with same keys as those initially passed to model.init).
                        Additionally, a total_loss key is added
                    list
                        Arbitrary number of predictions as np.array

                    """

            loss_vals = {}
            metric_vals = {}
            image_name_real = "real_images"
            image_name_fake = "fake_images"

            inputs = data_dict.pop('data')

            fake_images, discr_real, discr_fake, losses, *_ = model.run(inputs)

            for key, loss_val in losses.items():
                loss_vals[key] = loss_val

            for key, metric_fn in metrics.items():
                metric_vals[key + 'discr_real'] = metric_fn(
                    discr_real,
                    np.ones_like(discr_real))

            for key, metric_fn in metrics.items():
                metric_vals[key + 'discr_fake'] = metric_fn(
                    discr_fake,
                    np.zeros_like(discr_fake))

            for key, metric_fn in metrics.items():
                metric_vals[key + 'gen'] = metric_fn(
                    discr_fake,
                    np.ones_like(discr_fake))

            if model.training == False:
                # add prefix "val" in validation mode
                eval_loss_vals, eval_metrics_vals = {}, {}
                for key in loss_vals.keys():
                    eval_loss_vals["val_" + str(key)] = loss_vals[key]

                for key in metric_vals:
                    eval_metrics_vals["val_" + str(key)] = metric_vals[key]

                loss_vals = eval_loss_vals
                metric_vals = eval_metrics_vals

                image_name_real = "val_" + str(image_name_real)
                image_name_fake = "val_" + str(image_name_fake)

            for key, val in {**metric_vals, **loss_vals}.items():
                logging.info({"value": {"value": val.item(), "name": key,
                                        "env_appendix": "_%02d" % fold
                                        }})

            #logging.info({'image_grid': {"image_array": inputs, "name": image_name_real,
            #                             "title": "input_images", "env_appendix": "_%02d" % fold}})

            logging.info({'image_grid': {"image_array": fake_images, "name": image_name_fake,
                                         "title": "input_images", "env_appendix": "_%02d" % fold}})

            return metric_vals, loss_vals, [fake_images, discr_fake, discr_real]
