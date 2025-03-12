import torch
import torch.nn as nn
import snntorch as snn
from snntorch.functional import probe
import brevitas.nn as qnn  # Xilinx library for quantization
from brevitas.quant import Int8WeightPerTensorFixedPoint
from snntorch import utils
from snntorch import surrogate
import torch.nn.functional as F
from snntorch import functional as SF
from snntorch.functional import quant
from snntorch_network.RecurrentBlock import QuantRecurrentBlock
import numpy as np


class ExInbitoryNetwork(nn.Module):
    def __init__(self,
                num_inputs: int,
                num_hidden_1: int,
                num_hidden_2: int,
                num_outputs: int,

                grad: torch.autograd.Function,

                vth_in: float,
                vth_recurrent: float,
                vth_back: float,
                vth_out: float,

                beta_in: float,
                beta_recurrent: float,
                beta_back: float,
                beta_out: float,

                encoder_dim: int | None = None,
                vth_enc_value: float = 1.0,
                vth_std: float = 1000,
                beta_std: float = 1000,

                drop_recurrent: float = 0.0,
                drop_back: float = 0.0,
                drop_out: float = 0.0,

                state_quant: bool | object = False,

                time_dim: int = 1,

                num_bits: int = 8,
                layer_loss: object | None = None,
                weight_quant : object = Int8WeightPerTensorFixedPoint
                ):

        """

        Constructor for the ExInbitoryNetwork class.

        The vth and bias are the same for all neurons in each layer, except
        for the eventual encoding block (not shown in the diagram).

        Betas are the tension decays. There is no current decay, as we are
        using the leaky and not the synaptic model (in Lava, the LIF is by
        default "synaptic", so we set the current decay in Lava to zero).

        Some tests were performed in snnTorch using the synaptic model, but
        the training always set the value to the maximum. This plus the
        information from the paper 10.1109/JPROC.2023.3308088,  led to the
        decision to use the leaky model.

        The bias is disabled in all the layers to avoid an issue with the
        back layer, that would become saturated in time and result in always
        firing.
        Also, the bias in Lava / Loihi 2 is added to the voltage, after the
        current decay has happened.

        The initial value of the vth and beta of the encoding block are
        generated using a gaussian distribution.

        The quantization during training is important only for the constraints.
        Brevitas actually trains in float, but considers the bit number
        constraints.


        Block structure
              +-----------+     +-----------+    +------------+              +-----------+          +-----------+     +-----------+
        +--+  |           |     |           |    |            |  +--+        |           |          |           |     |           |    +--+
        |i |->|  In_Dense |---->|  In_LIF   |--->| In-R_Dense |--| -|------->|  F_LIF    |-------+->| Out_Dense |---->|  O_LIF    |--->|o |
        +--+  |           |     |           |    |            |  +--+        |           |       |  |           |     |           |    +--+
              +-----------+     +-----------+    +------------+   A          +-----------+       |  +-----------+     +-----------+
                                                                  |                              |
                                                                  |   +------+ +------+ +------+ |
                                                                  |   |      | |      | |      | |
                                                                  +---|Dense |<| ILIF |<|Dense |<+
                                                                      |      | |      | |      |
                                                                      +------+ +------+ +------+

        Parameters
        ----------

        num_inputs : int
            Number of channels in the input data, in other words, number of
            signals. For the accelerometer and gyroscope data, this is 6.

        num_hidden_1 : int
            Number of neurons in the In_LIF layer.

        num_hidden_2 : int
            Number of neurons in the F_LIF layer, which is the same as the
            ILIF layer

        num_outputs : int
            Number of neurons in the O_LIF layer. In other words the number of
            classes.

        grad : torch.autograd.Function
            Surrogate gradient function.

        vth_in : float
            Threshold of the In_LIF layer.

        vth_recurrent : float
            Threshold of the F_LIF layer.

        vth_back : float
            Threshold of the ILIF layer.

        vth_out : float
            Threshold of the O_LIF layer.

        beta_in : float
            Beta of the In_LIF layer.

        beta_recurrent : float
            Beta of the F_LIF layer.

        beta_back : float
            Beta of the ILIF layer.

        beta_out : float
            Beta of the O_LIF layer.

        encoder_dim : int | None
            Number of neurons in the eventual encoding block. If None, the
            encoding block is not used.

        vth_enc_value : float
            Maximum value to be used in the gaussian distribution that
            calculates the initialization value of the vth of the
            encoding block.

        vth_std : float
            Standard deviation of the gaussian distribution that calculates
            the initialization value of the vth of the encoding block.

        beta_std : float
            Standard deviation of the gaussian distribution that calculates
            the initialization value of the beta of the encoding block.
            Beta value is 1.

        drop_recurrent : float
            Dropout value before the F_LIF layer.

        drop_back : float
            Dropout value before the ILIF layer.

        drop_out : float
            Dropout value before the O_LIF layer.

        state_quant : False | snntorch.quant
            Function to be used for the quantization of neurons states. In
            practices, it is not used as the quantization is done in Lava.

        time_dim : int
            Position of the time dimension in the input data.

        num_bits : int
            Number of bits to be used in the quantization of the weights.

        layer_loss : object | None
            Loss function to be used in the training. If None, no loss is
            used.

        weight_quant : object
            Function to be used for the quantization of the weights. To be
            taken from the brevitas library.

        """

        super(ExInbitoryNetwork, self).__init__()
        self.layer_loss = layer_loss

        self.time_dim = time_dim

        self.quant = state_quant

        if encoder_dim is not None:
            # Create a gaussian distribution for the vth and beta, to cover
            # the full range of the encoder_dim

            self.encoder = True
            vth_e = self.gen_gaussian_distribution(encoder_dim, encoder_dim/2, vth_std, vth_enc_value)
            beta_e = self.gen_gaussian_distribution(encoder_dim, encoder_dim/2, beta_std)

            self.encoder_connection = qnn.QuantLinear(num_inputs, encoder_dim, bias=False,
                                            weight_bit_width=num_bits)
            #self.encoder_connection = nn.Linear(num_inputs, encoder_dim, bias=False)

            self.encoder_population = snn.Leaky(beta=beta_e,
                                    spike_grad=grad,
                                    threshold= vth_e,
                                    learn_threshold=True,
                                    learn_beta=True,
                                    reset_mechanism='zero',
                                    reset_delay=False)

            self.linear1 = qnn.QuantLinear(encoder_dim, num_hidden_1, bias=False,
                                       weight_bit_width=num_bits)
        else:
            self.encoder = False
            self.linear1 = qnn.QuantLinear(num_inputs, num_hidden_1, bias=False,
                                           weight_bit_width=num_bits, weight_quant=weight_quant)



        self.leaky1 = snn.Leaky(beta=beta_in,
                                spike_grad=grad,
                                threshold= vth_in,
                                learn_threshold=True,
                                learn_beta=True,
                                reset_mechanism='zero',
                                reset_delay=False,
                                state_quant=self.quant)

        self.linear2 = qnn.QuantLinear(num_hidden_1, num_hidden_2, bias=False,
                                        weight_bit_width=num_bits,
                                        weight_quant=  self.linear1.weight_quant)
        print(f"type of self.linear2 is {type(self.linear2)}")
        self.dropout_rec = nn.Dropout(p=drop_recurrent)

        self.recurrent = QuantRecurrentBlock(beta_recurrent, beta_back, vth_back, spike_grad=grad, linear_features = num_hidden_2,
                                            init_hidden=False, reset_delay=False, learn_beta=True,
                                            learn_threshold=True, learn_recurrent=True,
                                            vth=vth_recurrent, reset_mechanism="zero",
                                            shared_weight_quant=self.linear1.weight_quant,state_quant=self.quant, dropout=drop_back, output=True)
        self.linear3 = qnn.QuantLinear(num_hidden_2, num_outputs, bias=False,
                                                        weight_bit_width=num_bits,
                                                        weight_quant=  self.linear1.weight_quant)
        self.dropout_out = nn.Dropout(p=drop_out)

        self.leaky2 = snn.Leaky(beta=beta_out,
                                spike_grad=grad,
                                threshold= vth_out,
                                learn_threshold=True,
                                learn_beta=True,
                                reset_mechanism='zero',
                                reset_delay=False,
                                output=True,
                                state_quant=self.quant)
    def forward(self, data):
        spk_rec = []
        # In this dataset, we have the "continuous signal" already split into
        # segments of n time points. As the segments are presented in disorder
        # we need to reset the hidden states of the neurons in the network
        # for every segment.
        # utils.reset(self)  # resets hidden states for all LIF neurons in net

        if self.encoder:
            self.encoder_population.reset_hidden()

        self.leaky1.reset_hidden()
        self.recurrent.reset_hidden()
        self.leaky2.reset_hidden()
        rspk, rmem = self.recurrent.init_rleaky()

        if self.time_dim is not None:
            dims = list(range(data.dim()))  # Creates a list of dimensions
            #print(dims)
            dims.pop(self.time_dim)                     # Remove the selected dimension
            dims.insert(0, self.time_dim)               # Insert the selected dimension at the front
            data_permuted = data.permute(dims)  # Permute the tensor
        else:
            data_permuted = data

        if self.layer_loss is not None:

            if self.encoder:
                layer1_acc = []
                layer2_acc = []
                encoder_acc = []
            else:
                layer1_acc = []
                layer2_acc = []
        i = 0
        for slice in data_permuted:
            if self.encoder:

                x = self.encoder_connection(slice)
                x, _ = self.encoder_population(x)

                if self.layer_loss is not None:
                    encoder_acc.append(x.clone().cpu())


                x = self.linear1(x)
                x, _ = self.leaky1(x)

                if self.layer_loss is not None:
                    layer1_acc.append(x.clone().cpu())

            else:
                x = self.linear1(slice)
                x, _ = self.leaky1(x)

                if self.layer_loss is not None:
                    layer1_acc.append(x.clone().cpu())


            x = self.linear2(x)
            x = self.dropout_rec(x)

            rspk, rmem = self.recurrent(x, rspk, rmem)

            if self.layer_loss is not None:
                layer2_acc.append(rspk.clone().cpu())

            x = self.linear3(rspk)
            x = self.dropout_out(x)

            x, _ = self.leaky2(x)

            spk_rec.append(x)
        batch_out = torch.stack(spk_rec)

        if self.layer_loss is not None:
            if self.encoder:
                net_loss = self.layer_loss([torch.stack(layer1_acc), torch.stack(layer2_acc), torch.stack(encoder_acc)])
                del encoder_acc
            else:
                net_loss = self.layer_loss([torch.stack(layer1_acc), torch.stack(layer2_acc)])

            del layer1_acc
            del layer2_acc

            return batch_out, net_loss
        else:
            return batch_out

    def debug_init(self):

        self.spk_monitor = probe.OutputMonitor(self, instance = (snn.Leaky, snn.RLeaky))
        self.mem_monitor = probe.AttributeMonitor('mem', False, self, instance = (snn.Leaky))
        self.spk_monitor.enable()
        self.mem_monitor.enable()

    def debug_pause(self):
        self.spk_monitor.disable()
        self.mem_monitor.disable()

    def  debug_start(self):
        self.spk_monitor.enable()
        self.mem_monitor.enable()

    def clear_monitor(self):
        self.spk_monitor.clear_recorded_data()
        self.mem_monitor.clear_recorded_data()

    def get_monitor_results(self):
        return self.spk_monitor, self.mem_monitor



    @staticmethod
    def gen_gaussian_distribution( len, mean, std, max=1.0):

        #extend the binning in oder to create
        # the full gaussian and then cut the final part that goes to zeros
        bin = round(len*1.40)
        mean = round(mean*1.40)
        offset = round((bin - len)/2)
        x = torch.linspace(0, bin, bin)
        y = torch.exp(-((x - mean) ** 2) / (2 * std ** 2))*max #multiply for the actual treshold max
        yy = y[offset:-offset]
        yy[yy < 0.01] = 0.1
        return yy


    def save_to_npz(self, path: str):
        """Save network to file.

        Save the snntorch network into a compressed numpy file.

        Parameters
        ----------
        path : str
            The path to save the network to.
        """

        w_scale = self.linear1.quant_weight_scale().detach().cpu().numpy()
        w_zero_point = self.linear1.quant_weight_zero_point().detach().cpu().numpy()

        # For each weight we have the float and quantized versions.
        # The neuron parameters are quantized later in lava.
        linear1 = self.linear1.weight.data.detach().cpu().numpy()
        linear1_quant = self.linear1.quant_weight().int().detach().cpu().numpy()
        leaky1_betas = self.leaky1.beta.data.detach().cpu().numpy()
        leaky1_vth = self.leaky1.threshold.data.detach().cpu().numpy()
        linear2 = self.linear2.weight.data.detach().cpu().numpy()
        linear2_quant = self.linear2.quant_weight().int().detach().cpu().numpy()

        recurrent_betas = self.recurrent.beta.data.detach().cpu().numpy()
        recurrent_vth = self.recurrent.threshold.data.detach().cpu().numpy()
        input_dense, input_dense_quant, activation_betas, activation_vth, output_dense, output_dense_quant = self.recurrent.recurrent.to_npz()

        linear3 = self.linear3.weight.data.detach().cpu().numpy()
        linear3_quant = self.linear3.quant_weight().int().detach().cpu().numpy()
        leaky2_betas = self.leaky2.beta.data.detach().cpu().numpy()
        leaky2_vth = self.leaky2.threshold.data.detach().cpu().numpy()

        if self.encoder:
            encoder_connection = self.encoder_connection.weight.data.detach().cpu().numpy()
            encoder_population_betas = self.encoder_population.beta.data.detach().cpu().numpy()
            encoder_population_vth = self.encoder_population.threshold.data.detach().cpu().numpy()

            # np.savez_compressed(path,w_scale=w_scale, w_zero_point=w_zero_point, encoder_connection=encoder_connection, encoder_population_betas=encoder_population_betas,
            #                     encoder_population_vth=encoder_population_vth, linear1=linear1, leaky1_betas=leaky1_betas,
            #                     leaky1_vth=leaky1_vth, linear2=linear2, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
            #                     input_dense=input_dense, activation_betas=activation_betas,activation_vth=activation_vth, output_dense=output_dense,
            #                     linear3=linear3, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)
            np.savez_compressed(path, w_scale=w_scale, w_zero_point=w_zero_point, encoder_connection=encoder_connection, encoder_population_betas=encoder_population_betas,
                                encoder_population_vth=encoder_population_vth, linear1=linear1, linear1_quant=linear1_quant, leaky1_betas=leaky1_betas,
                                leaky1_vth=leaky1_vth, linear2=linear2, linear2_quant=linear2_quant, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
                                input_dense=input_dense, input_dense_quant=input_dense_quant, activation_betas=activation_betas, activation_vth=activation_vth,output_dense=output_dense,
                                output_dense_quant=output_dense_quant, linear3=linear3, linear3_quant=linear3_quant, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)
        else:
            # np.savez_compressed(path, w_scale=w_scale, w_zero_point=w_zero_point, linear1=linear1, leaky1_betas=leaky1_betas,
            #                     leaky1_vth=leaky1_vth, linear2=linear2, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
            #                     input_dense=input_dense, activation_betas=activation_betas, activation_vth=activation_vth,output_dense=output_dense,
            #                     linear3=linear3, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)

            np.savez_compressed(path, w_scale=w_scale, w_zero_point=w_zero_point, linear1=linear1, linear1_quant=linear1_quant, leaky1_betas=leaky1_betas,
                                leaky1_vth=leaky1_vth, linear2=linear2, linear2_quant=linear2_quant, recurrent_betas=recurrent_betas, recurrent_vth=recurrent_vth,
                                input_dense=input_dense, input_dense_quant=input_dense_quant, activation_betas=activation_betas, activation_vth=activation_vth,output_dense=output_dense,
                                output_dense_quant=output_dense_quant, linear3=linear3, linear3_quant=linear3_quant, leaky2_betas=leaky2_betas, leaky2_vth=leaky2_vth)

    def from_npz(self, path):

        data = np.load(path,allow_pickle=True)
        if self.encoder:
            self.encoder_connection.weight.data = torch.tensor(data['encoder_connection'])
            self.encoder_population.beta.data = torch.tensor(data['encoder_population_betas' ])
            self.encoder_population.threshold.data = torch.tensor(data['encoder_population_vth'])

        self.linear1.weight.data = torch.tensor(data['linear1'])
        self.leaky1.beta.data = torch.tensor(data['leaky1_betas'])
        self.leaky1.threshold.data = torch.tensor(data['leaky1_vth'])

        self.linear2.weight.data = torch.tensor(data['linear2'])

        self.recurrent.beta.data = torch.tensor(data['recurrent_betas'])
        self.recurrent.threshold.data = torch.tensor(data['recurrent_vth'])

        self.recurrent.recurrent.from_npz(data['input_dense'], data['activation_betas'],data['activation_vth'], data['output_dense'])

        self.linear3.weight.data = torch.tensor(data['linear3'])
        self.leaky2.beta.data = torch.tensor(data['leaky2_betas'])
        self.leaky2.threshold.data = torch.tensor(data['leaky2_vth'])

    def print_params(self):
        if self.encoder :
            print("Encoder Population")
            print(self.encoder_population.beta)
            print(self.encoder_population.threshold)

        print("Leaky 1")
        print(self.leaky1.beta)
        print(self.leaky1.threshold)
        print("Recurrent")
        print(self.recurrent.beta)
        print(self.recurrent.threshold)
        print("Leaky 2")
        print(self.leaky2.beta)
        print(self.leaky2.threshold)
