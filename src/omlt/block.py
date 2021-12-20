import warnings
import weakref

import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData, declare_custom_block

from .utils import _extract_var_data

# TODO: Update documentation
"""
This module defines the base class for implementing a custom block
within Pyomo based on input / output connections.

This module also contains the implementation of the OmltBlock class. This
class is used in combination with a formulation object and optionally
with a list of input variables and output variables corresponding to the inputs
and outputs of the neural network.
The formulation object is responsible for managing the construction and any refinement
or manipulation of the actual constraints.

Example 1:
    import tensorflow.keras as keras
    from pyoml.opt.neuralnet.keras_reader import load_keras_sequential
    from pyoml.opt import OmltBlock, FullSpaceContinuousFormulation

    nn = keras.models.load_model(keras_fname)
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    m.neural_net_block.build_formulation(FullSpaceContinuousFormulation(net))

    m.obj = pyo.Objective(expr=(m.neural_net_block.outputs[2]-4.0)**2)
    status = pyo.SolverFactory('ipopt').solve(m, tee=True)
    pyo.assert_optimal_termination(status)

"""

# TODO: Change inputs_lists to inputs_dict - good fix
# I think we can support both list and dict
@declare_custom_block(name="_BaseInputOutputBlock")
class _BaseInputOutputBlockData(_BlockData):
    def __init__(self, component):
        """
        Any block that inherits off of this must implement and call
        this __init__ with the passed component. This is to support
        interactions with Pyomo.
        """
        super(_BaseInputOutputBlockData, self).__init__(component)
        self.__input_indexes = None
        self.__output_indexes = None
        self.__inputs_list = None
        self.__outputs_list = None

    def _setup_inputs_outputs(
        self, *, input_indexes, output_indexes#, input_vars=None, output_vars=None
    ):
        """
        This function should be called by the derived class to setup the
        list of inputs and outputs for the input / output block.
        Parameters
        ----------
        n_inputs : int
            The number of inputs to the block
        n_outputs : int
            The number of outputs from the block
        input_vars : list or None
            The list of var data objects that correspond to the inputs.
            This list must match the order of inputs from 0 .. n_inputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.
            If set to None, then an indexed variable "inputs" is created on the
            automatically.
        output_vars :  list or None
            The list of var data objects that correspond to the outputs.
            This list must match the order of inputs from 0 .. n_outputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.
            If set to None, then an indexed variable "outputs" is created on the
            automatically.
        """
        self.__input_indexes = input_indexes
        self.__output_indexes = output_indexes
        if not input_indexes or not output_indexes:
            # todo: implement this check higher up in the class hierarchy to provide more contextual error msg
            raise ValueError(
                "_BaseInputOutputBlock must have at least one input and at least one output."
            )

        #create the input variables
        # if input_vars is None:
        self.inputs_set = pyo.Set(initialize=input_indexes)
        self.inputs = pyo.Var(self.inputs_set, initialize=0)
        self.__inputs_list = weakref.ref(self.inputs)

        #if output_vars is None:
        self.outputs_set = pyo.Set(initialize=output_indexes)
        self.outputs = pyo.Var(self.outputs_set, initialize=0)
        self.__outputs_list = weakref.ref(self.outputs)


    def _setup_input_bounds(self, inputs_list, input_bounds=None):
        if input_bounds:
            # set bounds using provided input_bounds
            for i, index in enumerate(inputs_list):
                var  = inputs_list[index]

                if var.lb == None:  # set lower bound to input_bounds value
                    var.setlb(input_bounds[i][0])
                else:
                    # throw warning if var.lb is more loose than input_bounds value
                    if var.lb < input_bounds[i][0]:
                        warnings.warning(
                            "Variable {} lower bound {} is less tight then network definition bound {}".format(
                                var, var.lb, input_bounds[i][0]
                            )
                        )

                if var.ub == None:
                    var.setub(input_bounds[i][1])
                else:
                    # throw warning if var.ub is more loose than input_bounds value
                    if var.ub > input_bounds[i][1]:
                        warnings.warning(
                            "Variable {} upper bound {} is less tight then network definition bound {}".format(
                                var, var.ub, input_bounds[i][1]
                            )
                        )


    #todo: query formulation for inputs and outputs
    @property
    def inputs_list(self):
        return self.__inputs_list()

    @property
    def outputs_list(self):
        return self.__outputs_list()

    @property
    def scaled_inputs_list(self):
        return self.__scaled_inputs_list()

    @property
    def scaled_outputs_list(self):
        return self.__scaled_outputs_list()


@declare_custom_block(name="OmltBlock")
class OmltBlockData(_BaseInputOutputBlockData):
    def __init__(self, component):
        super(OmltBlockData, self).__init__(component)
        self.__formulation = None
        #self.__scaling_object = None

    # TODO: input_vars needs to be a dict
    def build_formulation(self, formulation):#, input_vars=None, output_vars=None):
        """
        Call this method to construct the constraints (and possibly
        intermediate variables) necessary for the particular neural network
        formulation. The formulation object can be accessed later through the
        "formulation" attribute.

        Parameters
        ----------
        formulation : instance of PyomoFormulation
            see, for example, FullSpaceContinuousFormulation
        input_vars : list or None
            The list of var data objects that correspond to the inputs.
            This list must match the order of inputs from 0 .. n_inputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.

            If set to None, then an indexed variable "inputs" is created on the
            block automatically.
        output_vars :  list or None
            The list of var data objects that correspond to the outputs.
            This list must match the order of inputs from 0 .. n_outputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.

            If set to None, then an indexed variable "outputs" is created on the
            block automatically.
        """
        # call to the base class to define the inputs and the outputs

        # TODO: Do we want to validate formulation.input_indexes with input_vars here?
        # maybe formulation.check_input_vars(input_vars) or something?
        # super(OmltBlockData, self)._setup_inputs_outputs(
        #     input_indexes=list(formulation.input_indexes),
        #     output_indexes=list(formulation.output_indexes)
        #     # input_vars=input_vars,
        #     # output_vars=output_vars,
        # )

        # super(OmltBlockData, self)._setup_scaled_inputs_outputs(
        #     scaling_object=formulation.scaling_object,
        #     input_bounds=formulation.input_bounds,
        #     use_scaling_expressions=False,
        # )

        self.__formulation = formulation

        # tell the formulation that it is working on this block (self)
        self.__formulation._set_block(self)

        # tell the formulation object to construct the necessary models
        self.formulation._build_formulation()

    @property
    def formulation(self):
        """The formulation object used to construct the constraints (and possibly
        intermediate variables) necessary to represent the neural network in Pyomo
        """
        return self.__formulation

    # @property
    # def scaling_object(self):
    #     """Return an instance of the scaling object that supports the ScalingInterface"""
    #     return self.formulation.scaling_object

    # @property
    # def input_bounds(self):
    #     """Return a list of tuples containing lower and upper bounds of neural network inputs"""
    #     return self.formulation.input_bounds
