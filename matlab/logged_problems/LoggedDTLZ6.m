classdef LoggedDTLZ6 < DTLZ6
    %DTLZ6 variant that records every PlatEMO Evaluation call.
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ6(obj,varargin{:});
            log_platemo_evaluations(Population);
        end
    end
end
