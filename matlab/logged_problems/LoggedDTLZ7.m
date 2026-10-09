classdef LoggedDTLZ7 < DTLZ7
    %DTLZ7 variant that records every PlatEMO Evaluation call.
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ7(obj,varargin{:});
            log_platemo_evaluations(Population);
        end
    end
end
