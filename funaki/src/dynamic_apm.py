import tensorflow as tf
from tensorflow import linalg as tfl
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability import sts
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static

from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import (
    StructuralTimeSeries,
)

def build_apm_model(
    observed_time_series,
    design_apm,
    design_hca,
    apm_weights_prior=None,
    hca_weights_prior=None,
    drift_scale_prior=None,
):
    apm = SparseDynamicAPM(
        design_matrix=design_apm,
        drift_scale_prior=drift_scale_prior,
        initial_weights_prior=apm_weights_prior,
        name="apm",
    )

    if not hca_weights_prior:
        hca_weights_prior = tfd.MultivariateNormalDiag(
            scale_diag=0.01 * tf.ones([design_hca.shape[1]], dtype=tf.float32)
        )

    hca = PosConstrainedLinearRegression(
        design_matrix=design_hca, weights_prior=hca_weights_prior, name="hca"
    )

    model = sts.Sum([apm, hca], observed_time_series=observed_time_series)
    return model


def _zero_dimensional_mvndiag(dtype):
    """Build a zero-dimensional MVNDiag object."""
    dummy_mvndiag = tfd.MultivariateNormalDiag(scale_diag=tf.ones([0], dtype=dtype))
    dummy_mvndiag.covariance = lambda: dummy_mvndiag.variance()[..., tf.newaxis]
    return dummy_mvndiag

def _observe_timeseries_fn(timeseries):
    """Build an observation_noise_fn that observes a Tensor timeseries."""

    def observation_noise_fn(t):
        current_slice = timeseries[..., t, :]
        return tfd.MultivariateNormalDiag(
            loc=current_slice, scale_diag=tf.zeros_like(current_slice)
        )

    return observation_noise_fn

class PosConstrainedLinearRegression(sts.StructuralTimeSeries):
    """Formal representation of a linear regression from provided covariates with
    positivity constraint on weights. Positivy is achieved using `tfb.Softplus`
    bijector.

    This model defines a time series given by a linear combination of
    covariate time series provided in a design matrix:
    ```python
    observed_time_series = matmul(design_matrix, weights)
    ```
    The design matrix has shape `[num_timesteps, num_features]`. The weights
    are treated as an unknown random variable of size `[num_features]` (both
    components also support batch shape), and are integrated over using the same
    approximate inference tools as other model parameters, i.e., generally HMC or
    variational inference.
    This component does not itself include observation noise; it defines a
    deterministic distribution with mass at the point
    `matmul(design_matrix, weights)`. In practice, it should be combined with
    observation noise from another component such as `tfp.sts.Sum`, as
    demonstrated below.
    #### Examples
    Given `series1`, `series2` as `Tensors` each of shape `[num_timesteps]`
    representing covariate time series, we create a regression model that
    conditions on these covariates:
    ```python
    regression = tfp.sts.LinearRegression(
    design_matrix=tf.stack([series1, series2], axis=-1),
    weights_prior=tfd.Normal(loc=0., scale=1.))
    ```
    Here we've also demonstrated specifying a custom prior, using an informative
    `Normal(0., 1.)` prior instead of the default weakly-informative prior.
    As a more advanced application, we might use the design matrix to encode
    holiday effects. For example, suppose we are modeling data from the month of
    December. We can combine day-of-week seasonality with special effects for
    Christmas Eve (Dec 24), Christmas (Dec 25), and New Year's Eve (Dec 31),
    by constructing a design matrix with indicators for those dates.
    ```python
    holiday_indicators = np.zeros([31, 3])
    holiday_indicators[23, 0] = 1  # Christmas Eve
    holiday_indicators[24, 1] = 1  # Christmas Day
    holiday_indicators[30, 2] = 1  # New Year's Eve
    holidays = tfp.sts.LinearRegression(design_matrix=holiday_indicators,
                                      name='holidays')
    day_of_week = tfp.sts.Seasonal(num_seasons=7,
                                 observed_time_series=observed_time_series,
                                 name='day_of_week')
    model = tfp.sts.Sum(components=[holidays, seasonal],
                      observed_time_series=observed_time_series)
    ```
    Note that the `Sum` component in the above model also incorporates
    observation noise, with prior scale heuristically inferred from
    `observed_time_series`. In these examples, we've used a single design
    matrix, but batching is also supported. If the design matrix has batch
    shape, the default behavior constructs weights with matching batch shape,
    which will fit a separate regression for each design matrix. This can be
    overridden by passing an explicit weights prior with appropriate batch
    shape. For example, if each design matrix in a batch contains features with
    the same semantics (e.g., if they represent per-group or per-observation
    covariates), we might choose to share statistical strength by fitting a
    single weight vector that broadcasts across all design matrices:
    ```python
    design_matrix = get_batch_of_inputs()
    design_matrix.shape  # => concat([batch_shape, [num_timesteps, num_features]])
    # Construct a prior with batch shape `[]` and event shape `[num_features]`,
    # so that it describes a single vector of weights.
    weights_prior = tfd.Independent(
      tfd.StudentT(df=5,
                   loc=tf.zeros([num_features]),
                   scale=tf.ones([num_features])),
      reinterpreted_batch_ndims=1)
    linear_regression = LinearRegression(design_matrix=design_matrix,
                                       weights_prior=weights_prior)
    ```
    """

    def __init__(self, design_matrix, weights_prior=None, name=None):
        """Specify a linear regression model.
        Note: the statistical behavior of the regression is determined by
        the broadcasting behavior of the `weights` `Tensor`:
        * `weights_prior.batch_shape == []`: shares a single set of weights across
          all design matrices and observed time series. This may make sense if
          the features in each design matrix have the same semantics (e.g.,
          grouping observations by country, with per-country design matrices
          capturing the same set of national economic indicators per country).
        * `weights_prior.batch_shape == `design_matrix.batch_shape`: fits separate
          weights for each design matrix. If there are multiple observed time series
          for each design matrix, this shares statistical strength over those
          observations.
        * `weights_prior.batch_shape == `observed_time_series.batch_shape`: fits a
          separate regression for each individual time series.
        When modeling batches of time series, you should think carefully about
        which behavior makes sense, and specify `weights_prior` accordingly:
        the defaults may not do what you want!
        Args:
          design_matrix: float `Tensor` of shape `concat([batch_shape,
            [num_timesteps, num_features]])`. This may also optionally be
            an instance of `tf.linalg.LinearOperator`.
          weights_prior: `tfd.Distribution` representing a prior over the regression
            weights. Must have event shape `[num_features]` and batch shape
            broadcastable to the design matrix's `batch_shape`. Alternately,
            `event_shape` may be scalar (`[]`), in which case the prior is
            internally broadcast as `TransformedDistribution(weights_prior,
            tfb.Identity(), event_shape=[num_features],
            batch_shape=design_matrix.batch_shape)`. If `None`,
            defaults to `StudentT(df=5, loc=0., scale=10.)`, a weakly-informative
            prior loosely inspired by the [Stan prior choice recommendations](
            https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).
            Default value: `None`.
          name: the name of this model component.
            Default value: 'LinearRegression'.
        """
        with tf.compat.v1.name_scope(
            name, "PosConstrainedLinearRegression", values=[design_matrix]
        ) as name:

            if not isinstance(design_matrix, tfl.LinearOperator):
                design_matrix = tfl.LinearOperatorFullMatrix(
                    tf.convert_to_tensor(value=design_matrix, name="design_matrix"),
                    name="design_matrix_linop",
                )

            if tf.compat.dimension_value(design_matrix.shape[-1]) is not None:
                num_features = design_matrix.shape[-1]
            else:
                num_features = design_matrix.shape_tensor()[-1]

            # Default to a weakly-informative StudentT(df=5, 0., 10.) prior.
            if weights_prior is None:
                weights_prior = tfd.StudentT(
                    df=5,
                    loc=tf.zeros([], dtype=design_matrix.dtype),
                    scale=10 * tf.ones([], dtype=design_matrix.dtype),
                )

            # Sugar: if prior is static scalar, broadcast it to a default shape.
            if weights_prior.event_shape.ndims == 0:
                if design_matrix.batch_shape.is_fully_defined():
                    design_matrix_batch_shape_ = design_matrix.batch_shape
                else:
                    design_matrix_batch_shape_ = design_matrix.batch_shape_tensor()
                weights_prior = tfd.TransformedDistribution(
                    weights_prior,
                    bijector=tfb.Identity(),
                    batch_shape=design_matrix_batch_shape_,
                    event_shape=[num_features],
                )

            tf.debugging.assert_same_float_dtype([design_matrix, weights_prior])

            self._design_matrix = design_matrix

            super().__init__(
                parameters=[
                    Parameter("weights", weights_prior, tfb.Softplus()),
                ],
                latent_size=0,
                name=name,
            )

    @property
    def design_matrix(self):
        """LinearOperator representing the design matrix."""
        return self._design_matrix

    def _make_state_space_model(
        self, num_timesteps, param_map, initial_state_prior=None, initial_step=0
    ):

        weights = param_map["weights"]  # shape: [B, num_features]
        predicted_timeseries = self.design_matrix.matmul(weights[..., tf.newaxis])

        dtype = self.design_matrix.dtype

        # Since this model has `latent_size=0`, the latent prior and
        # transition model are dummy objects (zero-dimensional MVNs).
        dummy_mvndiag = _zero_dimensional_mvndiag(dtype)
        if initial_state_prior is None:
            initial_state_prior = dummy_mvndiag

        return tfd.LinearGaussianStateSpaceModel(
            num_timesteps=num_timesteps,
            transition_matrix=tf.zeros([0, 0], dtype=dtype),
            transition_noise=dummy_mvndiag,
            observation_matrix=tf.zeros([1, 0], dtype=dtype),
            observation_noise=_observe_timeseries_fn(predicted_timeseries),
            initial_state_prior=initial_state_prior,
            initial_step=initial_step,
        )

class SparseDynamicAPMStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
    """State space model for a dynamic linear regression model from provided covariates
    handling a sparse design matrix. A state space model (SSM) posits a set of latent
    (unobserved) variables that evolve over time with dynamics specified by a
    probabilistic transition model `p(z[t+1] | z[t])`. At each timestep, we observe a
    value sampled from an observation model conditioned on the current state,
    `p(x[t] | z[t])`. The special case where both the transition and observation models
    are Gaussians with mean specified as a linear function of the inputs, is known as a
    linear Gaussian state space model and supports tractable exact probabilistic
    calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
    details.
    The dynamic linear regression model is a special case of a linear Gaussian SSM
    and a generalization of typical (static) linear regression. The model
    represents regression `weights` with a latent state which evolves via a
    Gaussian random walk:
    ```
    weights[t] ~ Normal(weights[t-1], drift_scale)
    ```
    The latent state (the weights) has dimension `num_features`, while the
    parameters `drift_scale` and `observation_noise_scale` are each (a batch of)
    scalars. The batch shape of this `Distribution` is the broadcast batch shape
    of these parameters, the `initial_state_prior`, and the
    `design_matrix`. `num_features` is determined from the last dimension of
    `design_matrix` (equivalent to the number of columns in the design matrix in
    linear regression).
    #### Mathematical Details
    The dynamic linear regression model implements a
    `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size =
    num_features` and `observation_size = 1` following the transition model:
    ```
    transition_matrix = eye(num_features)
    transition_noise ~ Normal(0, diag([drift_scale]))
    ```
    which implements the evolution of `weights` described above. The observation
    model is:
    ```
    observation_matrix[t] = design_matrix[t]
    observation_noise ~ Normal(0, observation_noise_scale)
    ```
    #### Examples
    Given `series1`, `series2` as `Tensors` each of shape `[num_timesteps]`
    representing covariate time series, we create a dynamic regression model which
    conditions on these via the following:
    ```python
    dynamic_regression_ssm = DynamicLinearRegressionStateSpaceModel(
      num_timesteps=42,
      design_matrix=tf.stack([series1, series2], axis=-1),
      drift_scale=3.14,
      initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=[1., 2.]),
      observation_noise_scale=1.)
    y = dynamic_regression_ssm.sample()  # shape [42, 1]
    lp = dynamic_regression_ssm.log_prob(y)  # scalar
    ```
    Passing additional parameter and `initial_state_prior` dimensions constructs a
    batch of models, consider the following:
    ```python
    dynamic_regression_ssm = DynamicLinearRegressionStateSpaceModel(
      num_timesteps=42,
      design_matrix=tf.stack([series1, series2], axis=-1),
      drift_scale=[3.14, 1.],
      initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=[1., 2.]),
      observation_noise_scale=[1., 2.])
    y = dynamic_regression_ssm.sample(3)  # shape [3, 2, 42, 1]
    lp = dynamic_regression_ssm.log_prob(y)  # shape [3, 2]
    ```
    Which (effectively) constructs two independent state space models; the first
    with `drift_scale = 3.14` and `observation_noise_scale = 1.`, the second with
    `drift_scale = 1.` and `observation_noise_scale = 2.`. We then sample from
    each of the models three times and calculate the log probability of each of
    the samples under each of the models.
    Similarly, it is also possible to add batch dimensions via the
    `design_matrix`.
    """

    def __init__(
        self,
        num_timesteps,
        design_matrix,
        drift_scale,
        initial_state_prior,
        observation_noise_scale=0.0,
        initial_step=0,
        validate_args=False,
        allow_nan_stats=True,
        name=None,
    ):
        """State space model for a dynamic linear regression using a sparse representation
        for the  design matrix.
        Args:
          num_timesteps: Scalar `int` `Tensor` number of timesteps to model
            with this distribution.
          design_matrix: float `Tensor` of shape `concat([batch_shape,
            [num_timesteps, num_features]])`.
          drift_scale: Scalar (any additional dimensions are treated as batch
            dimensions) `float` `Tensor` indicating the standard deviation of the
            latent state transitions.
          initial_state_prior: instance of `tfd.MultivariateNormal`
            representing the prior distribution on latent states.  Must have
            event shape `[num_features]`.
          observation_noise_scale: Scalar (any additional dimensions are
            treated as batch dimensions) `float` `Tensor` indicating the standard
            deviation of the observation noise.
            Default value: `0.`.
          initial_step: scalar `int` `Tensor` specifying the starting timestep.
            Default value: `0`.
          validate_args: Python `bool`. Whether to validate input with asserts. If
            `validate_args` is `False`, and the inputs are invalid, correct behavior
            is not guaranteed.
            Default value: `False`.
          allow_nan_stats: Python `bool`. If `False`, raise an
            exception if a statistic (e.g. mean/mode/etc...) is undefined for any
            batch member. If `True`, batch members with valid parameters leading to
            undefined statistics will return NaN for this statistic.
            Default value: `True`.
          name: Python `str` name prefixed to ops created by this class.
            Default value: 'DynamicLinearRegressionStateSpaceModel'.
        """

        with tf.name_scope(name or "SparseDynamicAPMStateSpaceModel") as name:
            dtype = dtype_util.common_dtype(
                [design_matrix, drift_scale, initial_state_prior]
            )

            design_matrix = tf.sparse.from_dense(
                tf.convert_to_tensor(value=design_matrix, dtype=dtype),
                name="design_matrix",
            )

            drift_scale = tf.convert_to_tensor(
                value=drift_scale, name="drift_scale", dtype=dtype
            )

            observation_noise_scale = tf.convert_to_tensor(
                value=observation_noise_scale,
                name="observation_noise_scale",
                dtype=dtype,
            )

            num_features = prefer_static.shape(design_matrix)[-1]
            batch_shape = prefer_static.shape(drift_scale).tolist()

            def observation_matrix_fn(t):
                observation_matrix = tf.linalg.LinearOperatorFullMatrix(
                    tf.sparse.to_dense(
                        tf.sparse.slice(design_matrix, [t, 0], [1, num_features])
                    )[..., tf.newaxis, :],
                    name="observation_matrix",
                )
                return observation_matrix

            def transition_noise_fn(t):
                if len(batch_shape) > 0:
                    scale_mult_shape = batch_shape + [num_features]
                    temp = tf.sparse.to_dense(
                        tf.sparse.slice(design_matrix, [t, 0], [1, num_features])
                    )  # [..., tf.newaxis, :]
                    cond = tf.tile(temp, multiples=batch_shape + [1])
                else:
                    scale_mult_shape = [num_features]
                    cond = tf.sparse.to_dense(
                        tf.sparse.slice(design_matrix, [t, 0], [1, num_features])
                    )

                scale_mult = tf.where(
                    tf.logical_not(tf.equal(cond, 0)),
                    drift_scale[..., tf.newaxis]
                    * tf.ones(scale_mult_shape, dtype=dtype),
                    tf.zeros(scale_mult_shape, dtype=dtype),
                )

                #                 scale_mult = drift_scale[..., tf.newaxis] * tf.ones(scale_mult_shape, dtype=dtype) *\
                #                     tf.linalg.diag(tf.cast(tf.logical_not(tf.equal(cond, 0)), tf.float32))

                mvn = tfd.MultivariateNormalDiag(
                    scale_diag=scale_mult, name="transition_noise"
                )

                return mvn

            self._drift_scale = drift_scale
            self._observation_noise_scale = observation_noise_scale

            super().__init__(
                num_timesteps=num_timesteps,
                transition_matrix=tf.linalg.LinearOperatorIdentity(
                    num_rows=num_features, dtype=dtype, name="transition_matrix"
                ),
                transition_noise=transition_noise_fn,
                observation_matrix=observation_matrix_fn,
                observation_noise=tfd.MultivariateNormalDiag(
                    scale_diag=observation_noise_scale[..., tf.newaxis],
                    name="observation_noise",
                ),
                initial_state_prior=initial_state_prior,
                initial_step=initial_step,
                allow_nan_stats=allow_nan_stats,
                validate_args=validate_args,
                name=name,
            )

    @property
    def drift_scale(self):
        """Standard deviation of the drift in weights at each timestep."""
        return self._drift_scale

    @property
    def observation_noise_scale(self):
        """Standard deviation of the observation noise."""
        return self._observation_noise_scale


class SparseDynamicAPM(tfp.sts.StructuralTimeSeries):
    """Formal representation of a dynamic apm model with a sparse representation of
    the design matrix. This is really a dynamic regression model in disguise.
    The dynamic linear regression model is a special case of a linear Gaussian SSM
    and a generalization of typical (static) linear regression. The model
    represents regression `weights` with a latent state which evolves via a
    Gaussian random walk:
    ```
    weights[t] ~ Normal(weights[t-1], drift_scale)
    ```
    The latent state has dimension `num_features`, while the parameters
    `drift_scale` and `observation_noise_scale` are each (a batch of) scalars. The
    batch shape of this `Distribution` is the broadcast batch shape of these
    parameters, the `initial_state_prior`, and the `design_matrix`. `num_features`
    is determined from the last dimension of `design_matrix` (equivalent to the
    number of columns in the design matrix in linear regression).
    """

    def __init__(
        self,
        design_matrix,
        drift_scale_prior=None,
        initial_weights_prior=None,
        observed_time_series=None,
        name=None,
    ):
        """Specify a dynamic linear regression.
        Args:
          design_matrix: float `Tensor` of shape `concat([batch_shape,
            [num_timesteps, num_features]])`.
          drift_scale_prior: instance of `tfd.Distribution` specifying a prior on
            the `drift_scale` parameter. If `None`, a heuristic default prior is
            constructed based on the provided `observed_time_series`.
            Default value: `None`.
          initial_weights_prior: instance of `tfd.MultivariateNormal` representing
            the prior distribution on the latent states (the regression weights).
            Must have event shape `[num_features]`. If `None`, a weakly-informative
            Normal(0., 10.) prior is used.
            Default value: `None`.
          observed_time_series: `float` `Tensor` of shape `batch_shape + [T, 1]`
            (omitting the trailing unit dimension is also supported when `T > 1`),
            specifying an observed time series. Any priors not explicitly set will
            be given default values according to the scale of the observed time
            series (or batch of time series). May optionally be an instance of
            `tfp.sts.MaskedTimeSeries`, which includes a mask `Tensor` to specify
            timesteps with missing observations.
            Default value: `None`.
          name: Python `str` for the name of this component.
            Default value: 'DynamicLinearRegression'.
        """

        with tf.name_scope(name or "SparseDynamicAPM") as name:

            dtype = dtype_util.common_dtype(
                [design_matrix, drift_scale_prior, initial_weights_prior]
            )

            num_features = prefer_static.shape(design_matrix)[-1]

            # Default to a weakly-informative Normal(0., 10.) for the initital state
            if initial_weights_prior is None:
                initial_weights_prior = tfd.MultivariateNormalDiag(
                    scale_diag=0.001 * tf.ones([num_features], dtype=dtype)
                )

            # Heuristic default priors. Overriding these may dramatically
            # change inference performance and results.

            if observed_time_series is None:
                observed_stddev = tf.constant(1.0, dtype=dtype)
            else:
                _, observed_stddev, _ = sts_util.empirical_statistics(
                    observed_time_series
                )

            if drift_scale_prior is None:
                drift_scale_prior = tfd.LogNormal(
                    loc=tf.math.log(0.0001 * observed_stddev),
                    scale=0.0001,
                    name="drift_scale_prior",
                )

            self._initial_state_prior = initial_weights_prior
            self._design_matrix = design_matrix

            super().__init__(
                parameters=[
                    Parameter(
                        "drift_scale",
                        drift_scale_prior,
                        tfb.Chain(
                            [
                                tfb.Shift(0.0)(
                                    tfb.Scale(observed_stddev)
                                ),  # tfb.AffineScalar(scale=observed_stddev),
                                tfb.Softplus(),
                            ]
                        ),
                    )
                ],
                latent_size=num_features,
                name=name,
            )

    @property
    def initial_state_prior(self):
        """Prior distribution on the initial latent state (level and scale)."""
        return self._initial_state_prior

    @property
    def design_matrix(self):
        """Tensor representing the design matrix."""
        return self._design_matrix

    def _make_state_space_model(
        self, num_timesteps, param_map, initial_state_prior=None, initial_step=0
    ):

        if initial_state_prior is None:
            initial_state_prior = self.initial_state_prior

        return SparseDynamicAPMStateSpaceModel(
            num_timesteps=num_timesteps,
            design_matrix=self.design_matrix,
            initial_state_prior=initial_state_prior,
            initial_step=initial_step,
            **param_map
        )
