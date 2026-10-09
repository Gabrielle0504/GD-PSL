classdef LoggedDTLZ5 < DTLZ5
    %DTLZ5 variant that records every PlatEMO Evaluation call.
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ5(obj,varargin{:});
            log_platemo_evaluations(Population);
        end
    end
end
