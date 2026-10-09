classdef LoggedDTLZ1 < DTLZ1
    %DTLZ1 variant that records every PlatEMO Evaluation call.
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ1(obj,varargin{:});
            log_platemo_evaluations(Population);
        end
    end
end
