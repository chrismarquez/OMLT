import abc
import weakref


class _PyomoFormulationInterface(abc.ABC):
    """Base class interface for a Pyomo formulation object. This class
    is largely internal, and developers of new formulations should derive from
    :class:`pyoml.opt.neuralnet.PyomoFormulation`."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def _set_block(self, block):
        pass

    @property
    @abc.abstractmethod
    def block(self):
        pass

    @property
    @abc.abstractmethod
    def input_indexes(self):
        pass

    @property
    @abc.abstractmethod
    def output_indexes(self):
        pass

    @abc.abstractmethod
    def _build_formulation(self):
        pass


class _PyomoFormulation(_PyomoFormulationInterface):
    def __init__(self)#, network_structure):
        """This is a base class for different Pyomo formulations. To create a new
        formulation, inherit from this class and implement the build_formulation method. See
        :class:`pyoml.opt.neuralnet.FullSpaceContinuousFormulation` for an example."""
        super(_PyomoFormulation, self).__init__()
        
        #todo: I think network_definition should go on the subclass
        #self.__network_definition = network_structure
        #
        self.__input_bounds = None
        self.__scaling_object = None
        self.__block = None

    def _set_block(self, block):
        self.__block = weakref.ref(block)

    
    def _setup_scaled_inputs_outputs(self):
        
        if self.scaling_object == None:
            ##
            self.block.scaled_inputs = weakref.ref(self.inputs)
            self.block.scaled_outputs = weakref.ref(self.outputs)
            self.block.scaled_inputs_list = self.block.inputs_list
            self.block.scaled_outputs_list = self.block.outputs_list
            ##

            # self.__scaled_inputs_list = weakref.ref(self.inputs_list)
            # self.__scaled_outputs_list = weakref.ref(self.outputs_list)
            # self.scaled_inputs = weakref.ref(self.inputs)
            # self.scaled_outputs = weakref.ref(self.outputs)
            
            self._setup_input_bounds(self.inputs_list) #, input_bounds)

        else:
            # create pyomo variables for scaled and unscaled terms, input bounds are also scaled
            self.block.scaled_inputs_set = pyo.Set(initialize=list(self.inputs_list))
            self.block.scaled_inputs = pyo.Var(self.scaled_inputs_set, initialize=0)

            self.block.scaled_outputs_set = pyo.Set(initialize=list(self.outputs_list))
            self.block.scaled_outputs = pyo.Var(self.scaled_outputs_set, initialize=0)

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
    def block(self):
        """The underlying block containing the constraints / variables for this formulation."""
        return self.__block()

   
    #todo: input_indexes and output_indexes should be defined on subclass?
    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        network_inputs = list(self.__network_definition.input_nodes)
        assert len(network_inputs) == 1, 'Unsupported multiple network input variables'
        return network_inputs[0].input_indexes

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        network_outputs = list(self.__network_definition.output_nodes)
        assert len(network_outputs) == 1, 'Unsupported multiple network output variables'
        return network_outputs[0].output_indexes

    @property
    def scaling_object(self):
        """The scaling object used in the underlying network definition."""
        return self.__scaling_object

    @property
    def input_bounds(self):
        """Return a list of tuples containing lower and upper bounds of neural network inputs"""
        return self.__input_bounds

    @abc.abstractmethod
    def _build_formulation(self):
        """This method is called by the OmltBlock object to build the
        corresponding mathematical formulation of the model.
        See :class:`pyoml.opt.neuralnet.FullSpaceContinuousFormulation` for
        an example of an implementation.
        """
        pass

    # @property
    # def network_definition(self):
    #     """The object providing a definition of the network structure. Network
    #     definitions can be loaded from common training packages (e.g., see
    #     :func:`optml.io.keras_reader.load_keras_sequential`.) For a description
    #     of the network definition object, see
    #     :class:`pyoml.opt.network_definition.NetworkDefinition`"""
    #     return self.__network_definition