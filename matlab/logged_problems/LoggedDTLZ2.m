classdef LoggedDTLZ2 < DTLZ2
    %DTLZ2 variant that records every PlatEMO Evaluation call.
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ2(obj,varargin{:});
            log_platemo_evaluations(Population);
        end
    end
end
