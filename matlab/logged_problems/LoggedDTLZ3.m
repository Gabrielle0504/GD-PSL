classdef LoggedDTLZ3 < DTLZ3
    %DTLZ3 variant that records every PlatEMO Evaluation call.
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ3(obj,varargin{:});
            log_platemo_evaluations(Population);
        end
    end
end
