function ReferencePF = generate_reference_pf(problemName,M,D,sampleCount,platemoRoot)
%GENERATE_REFERENCE_PF Return PlatEMO's full-dimensional true PF sample.
% The result is projected by Python only after generation, matching the
% reference construction used by the Pang baseline.

    if nargin < 5
        platemoRoot = '';
    end
    if ~isempty(platemoRoot)
        if ~isfolder(platemoRoot)
            error('generate_reference_pf:InvalidRoot', ...
                  'PlatEMO root does not exist: %s', platemoRoot);
        end
        addpath(genpath(platemoRoot));
    end
    % PlatEMO class files use upper-case benchmark names.  This matters on
    % Linux even though MATLAB on Windows accepts the lower-case CLI names.
    requestedProblem = upper(char(problemName));

    rng(1,'twister');
    Problem = feval(requestedProblem,'M',double(M),'D',double(D));
    ReferencePF = double(Problem.GetOptimum(double(sampleCount)));
    ReferencePF = ReferencePF(all(isfinite(ReferencePF),2),:);
    ReferencePF = unique(ReferencePF,'rows','stable');
end
