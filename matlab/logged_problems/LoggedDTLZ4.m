classdef LoggedDTLZ4 < DTLZ4
    %DTLZ4 variant that records every PlatEMO Evaluation call.
    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@DTLZ4(obj,varargin{:});
            log_platemo_evaluations(Population);
        end
    end
end
