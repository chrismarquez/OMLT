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
        # else:
            # if isinstance(input_vars, list):
            #     #TODO: extract inputs_vars here into a pure list of var data objects
            #     if len(input_indexes) != len(input_vars):
            #         raise ValueError("The number of input variables does match the number of input indices")
            #     self.inputs = pyo.Reference(input_vars) #create a Pyomo `Reference` to passed-in variables. 
            #     #swap out the indices in the `Reference` object's `OrderedDict` to the input_indexes
            #     for (i,(k,v)) in enumerate(self.inputs._data.items()):
            #         idx = input_indexes[i]
            #         if len(idx) == 1:
            #             idx = idx[0]
            #         self.inputs._data[idx] = self.inputs._data.pop(k)
            #     self.__inputs_list = weakref.ref(self.inputs)
            #     #self.__inputs_list = input_vars
            
            # #the user must have passed in an IndexedVar
            # #TODO: We need to check that the input_vars indices are the same as __input_indexes
            # else:
            #     self.__inputs_list = weakref.ref(input_vars)
            # for index in self.__input_indexes:
            #     if len(index) == 1: #if the index is a tuple of one element
            #         if index[0] not in self.inputs:
            #             raise ValueError(f"Input index {index} not in IndexedVar {input_vars}")
            #     elif index not in self.inputs:#input_vars:
            #         raise ValueError(f"Input index {index} not in IndexedVar {input_vars}")

        #if output_vars is None:
        self.outputs_set = pyo.Set(initialize=output_indexes)
        self.outputs = pyo.Var(self.outputs_set, initialize=0)
        self.__outputs_list = weakref.ref(self.outputs)
        #else:
            # if isinstance(output_vars, list):
            #     if len(output_indexes) != len(output_vars):
            #         raise ValueError("The number of output variables does match the number of output indices")
            #     self.outputs = pyo.Reference(output_vars) #create a Pyomo `Reference` to passed-in variables. 
            #     for (i,(k,v)) in enumerate(self.outputs._data.items()):
            #         idx = output_indexes[i]
            #         if len(idx) == 1:
            #             idx = idx[0]
            #         self.outputs._data[idx] = self.outputs._data.pop(k)
            #     self.__outputs_list = weakref.ref(self.outputs)
            #     #self.__outputs_list = output_vars
            # else:
            #     self.__outputs_list = weakref.ref(output_vars)
            # for index in self.__output_indexes:
            #     if len(index) == 1: #if the index is a tuple of one element
            #         if index[0] not in self.outputs:
            #             raise ValueError(f"Output index {index} not in IndexedVar {output_vars}")
            #     elif index not in self.outputs:#output_vars:
            #         raise ValueError(f"Output index {index} not in IndexedVar {output_vars}")

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

    def _setup_scaled_inputs_outputs(
        self, *, scaling_object=None, input_bounds=None, use_scaling_expressions=False
    ):
        if scaling_object == None:
            self.__scaled_inputs_list = weakref.ref(self.inputs_list)
            self.__scaled_outputs_list = weakref.ref(self.outputs_list)
            self.scaled_inputs = weakref.ref(self.inputs)
            self.scaled_outputs = weakref.ref(self.outputs)
            self._setup_input_bounds(self.inputs_list, input_bounds)

        #TODO: decide how we want to handle this. this code never gets called. should we allow scaling using expressions?
        elif scaling_object and use_scaling_expressions:
            # use pyomo Expressions for scaled and unscaled terms, variable bounds are not directly captured
            self.__scaled_inputs_list = scaling_object.get_scaled_input_expressions(
                self.inputs_list
            )
            self.__scaled_outputs_list = scaling_object.get_scaled_output_expressions(
                self.outputs_list
            )
            # Bounds only set on unscaled inputs
            self._setup_input_bounds(self.inputs_list, input_bounds)

        else:
            # create pyomo variables for scaled and unscaled terms, input bounds are also scaled
            self.scaled_inputs_set = pyo.Set(initialize=list(self.inputs_list))
            self.scaled_inputs = pyo.Var(self.scaled_inputs_set, initialize=0)

            self.scaled_outputs_set = pyo.Set(initialize=list(self.outputs_list))
            self.scaled_outputs = pyo.Var(self.scaled_outputs_set, initialize=0)

            # set scaled variables lists
            self.__scaled_inputs_list = weakref.ref(self.scaled_inputs)
            self.__scaled_outputs_list = weakref.ref(self.scaled_outputs)

            # Create constraints connecting scaled and unscaled variables
            self.__scale_input_con = pyo.Constraint(self.scaled_inputs_set)
            self.__unscale_output_con = pyo.Constraint(self.scaled_outputs_set)

            # ToDo: decide if scaling expression should be indexed with the same
            # index as variables
            scaled_input_expressions = scaling_object.get_scaled_input_expressions(
                self.inputs_list
            )
            unscaled_output_expressions = (
                scaling_object.get_unscaled_output_expressions(self.scaled_outputs_list)
            )

            # scaled input constraints
            for i in range(len(self.scaled_inputs_set)):
                self.__scale_input_con[i] = (
                    self.scaled_inputs[i] == scaled_input_expressions[i]
                )
            # unscaled output constraints
            for i in range(len(self.scaled_outputs_set)):
                self.__unscale_output_con[i] = (
                    self.outputs_list[i] == unscaled_output_expressions[i]
                )

            # scale input bounds
            if input_bounds:
                input_lower = [input_bounds[i][0] for i in range(len(input_bounds))]
                input_upper = [input_bounds[i][1] for i in range(len(input_bounds))]
                scaled_lower = scaling_object.get_scaled_input_expressions(input_lower)
                scaled_upper = scaling_object.get_scaled_input_expressions(input_upper)
                scaled_input_bounds = list(zip(scaled_lower, scaled_upper))
                self._setup_input_bounds(self.scaled_inputs_list, scaled_input_bounds)

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
        self.__scaling_object = None

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
        super(OmltBlockData, self)._setup_inputs_outputs(
            input_indexes=list(formulation.input_indexes),
            output_indexes=list(formulation.output_indexes)
            # input_vars=input_vars,
            # output_vars=output_vars,
        )

        super(OmltBlockData, self)._setup_scaled_inputs_outputs(
            scaling_object=formulation.scaling_object,
            input_bounds=formulation.input_bounds,
            use_scaling_expressions=False,
        )

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

    @property
    def scaling_object(self):
        """Return an instance of the scaling object that supports the ScalingInterface"""
        return self.formulation.scaling_object

    @property
    def input_bounds(self):
        """Return a list of tuples containing lower and upper bounds of neural network inputs"""
        return self.formulation.input_bounds
