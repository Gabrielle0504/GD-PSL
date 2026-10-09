classdef PangMOEAD < ALGORITHM
% <2023> <multi/many> <real/integer/label/binary/permutation>
% MOEA/D with Pang--Nan--Ishibuchi random weight perturbation.
% The implementation follows the strategy in Section III-B/IV of:
% L. M. Pang, Y. Nan, and H. Ishibuchi, IEEE SMC 2023.

    methods
        function main(Algorithm,Problem)
            %% Parameters and the fixed MOEA/D decomposition
            type = Algorithm.ParameterSet(1);
            [WOriginal,Problem.N] = UniformPoint(Problem.N,Problem.M);
            W = WOriginal;
            T = ceil(Problem.N/10);
            B = PangMOEAD.Neighbours(W,T);

            Population = Problem.Initialization();
            Z = min(Population.objs,[],1);
            generation = 0;
            perturbationPeriod = 100;
            if Problem.M <= 3
                perturbationMagnitude = 0.1;
            elseif Problem.M >= 8
                perturbationMagnitude = 0.3;
            else
                perturbationMagnitude = 0.2;
            end

            %% Search and archive logging are handled by PlatEMO's problem
            while Algorithm.NotTerminated(Population)
                generation = generation + 1;
                if mod(generation,perturbationPeriod) == 0
                    delta = -perturbationMagnitude + 2*perturbationMagnitude*rand(size(WOriginal));
                    % The authors perturb the original uniform weights at
                    % every event; perturbations never accumulate.
                    W = max(WOriginal + delta,1e-6);
                    W = W./sum(W,2);
                    B = PangMOEAD.Neighbours(W,T);
                end

                for i = 1 : Problem.N
                    % UniformPoint can revise N, so stop the final partial
                    % generation exactly at the shared evaluation budget.
                    if Problem.FE >= Problem.maxFE
                        break;
                    end
                    P = B(i,randperm(size(B,2)));
                    Offspring = OperatorGAhalf(Problem,Population(P(1:2)));
                    Z = min(Z,Offspring.obj);

                    if type ~= 1
                        error('PangMOEAD:UnsupportedType', ...
                              'The authors use only the PBI aggregation function.');
                    end
                    normW   = sqrt(sum(W(P,:).^2,2));
                    normP   = sqrt(sum((Population(P).objs-repmat(Z,T,1)).^2,2));
                    normO   = sqrt(sum((Offspring.obj-Z).^2,2));
                    CosineP = sum((Population(P).objs-repmat(Z,T,1)).*W(P,:),2)./normW./max(normP,1e-12);
                    CosineO = sum(repmat(Offspring.obj-Z,T,1).*W(P,:),2)./normW./max(normO,1e-12);
                    g_old   = normP.*CosineP + 5*normP.*sqrt(max(0,1-CosineP.^2));
                    g_new   = normO.*CosineO + 5*normO.*sqrt(max(0,1-CosineO.^2));
                    Population(P(g_old>=g_new)) = Offspring;
                end
            end
        end
    end

    methods (Static, Access = private)
        function B = Neighbours(W,T)
            distances = pdist2(W,W);
            [~,B] = sort(distances,2);
            B = B(:,1:T);
        end
    end
end
