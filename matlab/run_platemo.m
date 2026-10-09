function [X, F, C, HistoryX, HistoryF, HistoryFE, HistoryTime] = run_platemo(algorithmName, problemName, N, maxFE, M, D, platemoRoot, algorithmParameters, randomSeed)
%RUN_PLATEMO Generic bridge from Python to a PlatEMO algorithm/problem pair.

    if nargin < 9
        randomSeed = 1;
    end
    if nargin < 8
        algorithmParameters = [];
    end
    if nargin < 7
        platemoRoot = '';
    end

    if ~isempty(platemoRoot)
        if ~isfolder(platemoRoot)
            error('run_platemo:InvalidRoot', 'PlatEMO root does not exist: %s', platemoRoot);
        end
        addpath(genpath(platemoRoot));
    end
    bridgeDir = fileparts(mfilename('fullpath'));
    loggedProblemDir = fullfile(bridgeDir, 'logged_problems');
    if isfolder(loggedProblemDir)
        addpath(genpath(loggedProblemDir));
    end
    % The project ships the seven RE definitions used by the formal suite.
    % Add them after PlatEMO so the optimizer remains PlatEMO's implementation.
    customProblemDir = fullfile(bridgeDir, 're_problems');
    if isfolder(customProblemDir)
        addpath(genpath(customProblemDir));
    end
    if isempty(which('platemo'))
        error('run_platemo:MissingPlatEMO', ...
              'platemo.m is not on the MATLAB path. Set --platemo-root or PLATEMO_ROOT.');
    end

    rng(double(randomSeed), 'twister');

    requestedAlgorithm = upper(char(algorithmName));
    % Project-local adapters preserve each algorithm's operators and selection
    % logic while shortening only the final offspring batch at maxFE.
    if strcmp(requestedAlgorithm, 'NSGAII')
        algorithmHandle = @NSGAIIExactBudget;
    elseif strcmp(requestedAlgorithm, 'MOEAD')
        algorithmHandle = @MOEADExactBudget;
    elseif strcmp(requestedAlgorithm, 'NSGAIII')
        algorithmHandle = @NSGAIIIExactBudget;
    else
        algorithmHandle = str2func(char(algorithmName));
    end
    requestedProblem = upper(char(problemName));
    if strcmp(requestedProblem, 'DTLZ2')
        problemHandle = @LoggedDTLZ2;
    elseif strcmp(requestedProblem, 'DTLZ7')
        problemHandle = @LoggedDTLZ7;
    else
        problemHandle = str2func(char(problemName));
    end
    if isempty(which(func2str(algorithmHandle)))
        error('run_platemo:UnknownAlgorithm', 'Unknown PlatEMO algorithm: %s', algorithmName);
    end
    if isempty(which(func2str(problemHandle)))
        error('run_platemo:UnknownProblem', 'Unknown PlatEMO problem: %s', problemName);
    end

    if isempty(algorithmParameters)
        algorithmSpec = algorithmHandle;
    else
        parameterCells = num2cell(double(algorithmParameters(:)'));
        algorithmSpec = [{algorithmHandle}, parameterCells];
    end

    global APS_EVAL_DECS APS_EVAL_OBJS APS_EVAL_CONS APS_EVAL_FE APS_EVAL_COUNT APS_EVAL_TIME APS_EVAL_START;
    APS_EVAL_DECS = zeros(0, double(D));
    APS_EVAL_OBJS = zeros(0, double(M));
    APS_EVAL_CONS = zeros(0, 0);
    APS_EVAL_FE = zeros(0, 1);
    APS_EVAL_COUNT = 0;
    APS_EVAL_TIME = zeros(0, 1);
    APS_EVAL_START = tic;

    [X, F, C] = platemo( ...
        'algorithm', algorithmSpec, ...
        'problem', problemHandle, ...
        'N', double(N), ...
        'maxFE', double(maxFE), ...
        'M', double(M), ...
        'D', double(D), ...
        'draw', false);
    HistoryX = APS_EVAL_DECS;
    HistoryF = APS_EVAL_OBJS;
    HistoryFE = APS_EVAL_FE;
    HistoryTime = APS_EVAL_TIME;
end
