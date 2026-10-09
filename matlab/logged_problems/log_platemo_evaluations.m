function log_platemo_evaluations(Population)
%LOG_PLATEMO_EVALUATIONS Append one true PlatEMO evaluation batch to globals.
    global APS_EVAL_DECS APS_EVAL_OBJS APS_EVAL_CONS APS_EVAL_FE APS_EVAL_COUNT APS_EVAL_TIME APS_EVAL_START;
    if isempty(APS_EVAL_COUNT)
        APS_EVAL_COUNT = 0;
    end
    n = length(Population);
    if n == 0
        return;
    end
    APS_EVAL_DECS = [APS_EVAL_DECS; Population.decs]; %#ok<AGROW>
    APS_EVAL_OBJS = [APS_EVAL_OBJS; Population.objs]; %#ok<AGROW>
    APS_EVAL_CONS = [APS_EVAL_CONS; Population.cons]; %#ok<AGROW>
    APS_EVAL_FE = [APS_EVAL_FE; (APS_EVAL_COUNT + (1:n))']; %#ok<AGROW>
    APS_EVAL_TIME = [APS_EVAL_TIME; repmat(toc(APS_EVAL_START), n, 1)]; %#ok<AGROW>
    APS_EVAL_COUNT = APS_EVAL_COUNT + n;
end
