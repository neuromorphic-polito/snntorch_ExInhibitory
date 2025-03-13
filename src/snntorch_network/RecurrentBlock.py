"""RLeaky snntroch recurrent block extended with an inhibitory recurrent
connection with a neuron population"""
import torch
import torch.nn as nn

import snntorch as snn
from snntorch.functional import quant

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint


class QuantRecurrentBlock(snn.RLeaky):
    """RLeaky snntroch recurrent block extended with an inhibitory recurrent
       connection with a neuron population"""

    def __init__(
        self,

        # Parameters for the ILIF class
        back_beta: float,
        back_vth: float,


        # Parameters for the RLeaky class
        beta: float,
        linear_features=None,
        kernel_size=None,
        vth: float = 1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        learn_recurrent=True,
        reset_mechanism="zero",
        state_quant=False,
        output=False,
        reset_delay=False,

        # Default parameters for the ILIF class
        back_grad=None,
        dropout=0.0,

        # Other parameters
        shared_weight_quant: object = Int8WeightPerTensorFixedPoint,

    ):

        """Constructor for the QuantRecurrentBlock class.


                    +-----------+
        +--+        |           |
     ---| -|------->|  F_LIF    |-------+->
        +--+        |           |       |
        A          +-----------+       |
        |                              |
        |   +------+ +------+ +------+ |
        |   |      | |      | |      | |
        +---|Dense |<| ILIF |<|Dense |<+
            |      | |      | |      |
            +------+ +------+ +------+

        For the description of the parameters that are passed directly to the
        RLeaky class, see the RLeaky class documentation.

        Parameters
        ----------

        back_beta: float
            Beta of the ILIF neuron

        back_vth: float
            Threshold of the ILIF neuron


        back_grad : object
            Function to be used for the gradient of the ILIF neuron.

        dropout : float
            Dropout value before the ILIF layer.

        shared_weight_quant : object
            Function to be used for the quantization of the weights. To be
            taken from the brevitas library.
        """

        super(QuantRecurrentBlock, self).__init__(
            beta=beta,
            all_to_all=True,
            linear_features=linear_features,
            kernel_size=kernel_size,
            threshold=vth,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            inhibition=inhibition,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            learn_recurrent=learn_recurrent,
            reset_mechanism=reset_mechanism,
            state_quant=state_quant,
            output=output,
            reset_delay=reset_delay
        )

        assert linear_features is not None, "Linear features must not be None"

        self.back_grad = back_grad if back_grad is not None else spike_grad

        # Check if we are quantizing the state
        if not state_quant:
            self.back_quant = False
        else:
            self.back_quant = state_quant

        self.back_beta = back_beta
        self.back_vth = back_vth
        self.dropout = dropout
        self.shared_weight_quant = shared_weight_quant
        self.overwrite_self_recurrent()

    def overwrite_self_recurrent(self):
        """Overwrite the recurrent layer of the RLeaky class with a new
        QuantInibitoryBlock layer.
        """
        self.recurrent = QuantInibitoryBlock(
            self.back_beta,
            self.back_vth,
            self.back_grad,
            self.linear_features,
            shared_weight_quant=self.shared_weight_quant,
            state_quant=self.back_quant,
            dropout=self.dropout)

    def _build_state_function_hidden(self, input_):
        """We are redefining the torch version for the parent (RLeaky).

        The original torch version, in the == 1 condition does the following:

        state_fn = \
            self._base_state_function_hidden(input_) - \
            self._base_state_function_hidden(input_) * self.reset

        There are two calls to the _base_state_function_hidden, which
        apparently does not generate any issues for the RLeaky implementation,
        as the recurrent connection is only a Dense layer.

        Instead in our case we have a neuron population that updates the state
        twice for each pass (and maybe also does back propagation twice).

        The main issue is that the neuron updates twice its state, so it is
        like each input is presented twice.
        """
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function_hidden(input_)
                - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = (1.0 - self.reset) * \
                self._base_state_function_hidden(input_)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn

    def _build_state_function(self, input_, spk, mem):
        """We are redefining the torch version for the parent (RLeaky).

        The original torch version, in the == 1 condition does the following:

        state_fn = \
            self._base_state_function(input_, spk, mem) -
            self._base_state_function(input_, spk, mem) * self.reset

        There are two calls to the _base_state_function_hidden, which
        apparently does not generate any issues for the RLeaky implementation,
        as the recurrent connection is only a Dense layer.

        Instead in our case we have a neuron population that updates the state
        twice for each pass (and maybe also does back propagation twice).

        The main issue is that the neuron updates twice its state, so it is
        like each input is presented twice.
        """

        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = self._base_state_function(
                input_, spk, mem - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = (1.0 - self.reset) * self._base_state_function(
                input_, spk, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, spk, mem)

        return state_fn

    def _base_state_function_hidden(self, input_):
        """We are redefining the torch version for the parent (RLeaky).
        The original torch version is as follows:

            base_fn = (
                self.beta.clamp(0, 1) * self.mem
                + input_
                + self.recurrent(self.spk)
            )
        The original recurrent connection is not inhibitory, as we need it.
        To change to inhibitory, we simply change the recurrent sign to
        negative.
        """
        base_fn = (
            self.beta.clamp(0, 1) * self.mem
            + input_
            - self.recurrent(self.spk)
        )
        return base_fn

    def _base_state_function(self, input_, spk, mem):
        """We are redefining the torch version for the parent (RLeaky).
        The original torch version is as follows:

            base_fn = \
                self.beta.clamp(0, 1) * mem + input_ + self.recurrent(spk)

        The original recurrent connection is not inhibitory, as we need it.
        To change to inhibitory, we simply change the recurrent sign to
        negative.
        """

        base_fn = \
            self.beta.clamp(0, 1) * mem + input_ - self.recurrent(spk)
        return base_fn

    def reset_hidden(self):
        """We need to add the reset for the recurrent connection, because in
        the RLeaky implementation there is no neuron in this connection."""

        super().reset_hidden()
        self.recurrent.reset_hidden()


class QuantInibitoryBlock(nn.Module):
    def __init__(self,
                 beta: float,
                 vth: float,
                 grad: object,
                 features: int,
                 shared_weight_quant: object,
                 num_bits: int = 8,
                 state_quant: object = False,
                 dropout: float = 0.0,
                 delay: bool = False) -> None:
        """Constructor of the QuantInibitoryBlock class

         +------+ +------+ +------+
         |      | |      | |      |
        -|Dense |<| ILIF |<|Dense |<
         |      | |      | |      |
         +------+ +------+ +------+

        Parameters
        ----------
        beta : float
            beta of the ILIF layer

        vth : float
            vth of the ILIF layer

        grad : torch.autograd.Function
            Surrogate gradient function.

        features : int
            Number of neurons in the ILIF layer

        shared_weight_quant :  object
            Function to be used for the quantization of the weights. To be
            taken from the brevitas library.

        num_bits : int, optional
            Number of bits for the quantization of the weights, by default 8

        state_quant : object, optional
            Function to be used to quantize the states, by default False

        dropout : float, optional
            Dropout value before the ILIF layer, by default 0.0

        delay : bool, optional
            Decide to apply delay to emulate Lava / Loihi behavior,
            by default False, network is robust enough to work without it
        """
        super(QuantInibitoryBlock, self).__init__()

        # For the connections, we use the brevitas objects, that will help
        # us in the quantization

        # Dense in the right
        self.input_dense = qnn.QuantLinear(
            in_features=features,
            out_features=features,
            bias=False,
            weight_quant=shared_weight_quant,
            weight_bit_width=num_bits
            )

        # The parameter is the probability of dropping a connection.
        self.dropout = nn.Dropout(dropout)

        # Not sure if this is beneficial or not, or if it can be removed
        # because it has no effect.
        self.activation_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint,
            act_bit_width=24,
            return_quant_tensor=True
            )

        if not state_quant:
            self.activation = snn.Leaky(
                beta=beta,
                spike_grad=grad,
                threshold=vth,
                learn_threshold=True,
                learn_beta=True,
                reset_mechanism='zero',
                reset_delay=False
                )
        else:
            # This quant is not from brevitas, but from snntorch
            activativation_quant = quant.state_quant(
                num_bits=16,
                threshold=vth
                )

            # reset delay to False so the behavior is the same as in Loihi 2
            # and spinnaker
            self.activation = snn.Leaky(
                beta=beta,
                spike_grad=grad,
                threshold=vth,
                learn_threshold=True,
                learn_beta=True,
                reset_mechanism='zero',
                reset_delay=False,
                state_quant=activativation_quant
                )

        # Dense to the left
        self.output_dense = qnn.QuantLinear(
            in_features=features,
            out_features=features,
            bias=False,
            weight_quant=shared_weight_quant,
            weight_bit_width=num_bits
            )

        # Not sure if this is beneficial or not, or if it can be removed
        # because it has no effect.
        self.out_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFixedPoint,
            act_bit_width=24,
            return_quant_tensor=True
            )

        # Because in Lava / Loihi 2 the recurrent branch has a delay, we can
        # set it here to emulate it
        self.delay: bool = delay
        if self.delay:
            self.features: int = features
            self.accumulator: torch.Tensor = torch.zeros((1, features))
        else:
            self.features = None

    def forward(self, input):
        # Add your forward pass implementation here
        if self.delay:
            # assigns x to the previous accumulator value, and save the new
            # input
            x: torch.Tensor = self.input_dense(self.accumulator.to(
                input.device))
            self.accumulator = input
        else:
            x = self.input_dense(input)

        x = self.dropout(x)
        x = self.activation_quant(x)

        # _ is the membrane potential. As we are using init_hidden to False,
        # it is returned by snntorch even if we are not using it.
        spk , _ = self.activation(x)

        out = self.output_dense(spk)

        out = self.out_quant(out)

        return out

    def reset_hidden(self) -> None:
        """Reset accumulator (if used) and the activation state"""

        if self.delay:
            self.accumulator = torch.zeros((1, self.features))
        self.activation.reset_hidden()

    def to_npz(self):

        # Weights, float and quantized
        input_dense = self.input_dense.weight.detach().cpu().numpy()
        input_dense_quant = self.input_dense.quant_weight().value.detach().cpu().numpy()

        # Neuron state
        activation_beta = self.activation.beta.data.detach().cpu().numpy()
        activation_vth = self.activation.threshold.data.detach().cpu().numpy()

        # Weights, float and quantized
        output_dense = self.output_dense.weight.detach().cpu().numpy()
        output_dense_quant = self.output_dense.quant_weight().value.detach().cpu().numpy()

        return input_dense,input_dense_quant, \
                activation_beta, activation_vth, \
                output_dense, output_dense_quant

    def from_npz(self, input_dense, activation_beta, activation_vth, output_dense):

        self.input_dense.weight.data = torch.tensor(input_dense)
        self.activation.beta.data = torch.tensor(activation_beta)
        self.activation.threshold.data = torch.tensor(activation_vth)
        self.output_dense.weight.data = torch.tensor(output_dense)
