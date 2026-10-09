classdef REProblemBase < PROBLEM
    %REPROBLEMBASE PlatEMO-compatible implementations of the project RE set.
    %
    % The Python definitions use constraint violation as the final objective
    % for the constrained RE benchmarks.  This class preserves that contract
    % so the MATLAB search output can be validated by the Python pipeline.

    methods
        function Population = Evaluation(obj,varargin)
            Population = Evaluation@PROBLEM(obj,varargin{:});
            log_platemo_evaluations(Population);
        end

        function Setting(obj)
            name = upper(class(obj));
            switch name
                case 'RE21'
                    obj.M = 2; obj.D = 4;
                    obj.lower = [1 sqrt(2) sqrt(2) 1];
                    obj.upper = [3 3 3 3];
                case 'RE24'
                    obj.M = 2; obj.D = 2;
                    obj.lower = [0.5 0.5]; obj.upper = [4 50];
                case 'RE31'
                    obj.M = 3; obj.D = 3;
                    obj.lower = [1e-5 1e-5 1]; obj.upper = [100 100 3];
                case 'RE32'
                    obj.M = 3; obj.D = 4;
                    obj.lower = [0.125 0.1 0.1 0.125]; obj.upper = [5 10 10 5];
                case 'RE34'
                    obj.M = 3; obj.D = 5;
                    obj.lower = ones(1,5); obj.upper = 3*ones(1,5);
                case 'RE35'
                    obj.M = 3; obj.D = 7;
                    obj.lower = [2.6 0.7 17 7.3 7.3 2.9 5];
                    obj.upper = [3.6 0.8 28 8.3 8.3 3.9 5.5];
                case 'RE37'
                    obj.M = 3; obj.D = 4;
                    obj.lower = zeros(1,4); obj.upper = ones(1,4);
                otherwise
                    error('REProblemBase:UnknownProblem','Unsupported RE class %s.',name);
            end
            obj.encoding = ones(1,obj.D);
        end

        function PopObj = CalObj(obj,PopDec)
            x = double(PopDec);
            name = upper(class(obj));
            switch name
                case 'RE21', PopObj = obj.re21(x);
                case 'RE24', PopObj = obj.re24(x);
                case 'RE31', PopObj = obj.re31(x);
                case 'RE32', PopObj = obj.re32(x);
                case 'RE34', PopObj = obj.re34(x);
                case 'RE35', PopObj = obj.re35(x);
                case 'RE37', PopObj = obj.re37(x);
                otherwise, error('REProblemBase:UnknownProblem','Unsupported RE class %s.',name);
            end
        end
    end

    methods (Access=private)
        function v = violation(~, varargin)
            G = cat(2,varargin{:});
            v = sum(max(0,-G),2);
        end

        function y = nearest(~, values, x)
            values = values(:)';
            x = x(:);
            [~,idx] = min(abs(x-values),[],2);
            y = values(idx(:));
            y = y(:);
        end

        function F = re21(obj,x)
            F = 10; E = 2e5; L = 200;
            f1 = L*(2*x(:,1)+sqrt(2)*x(:,2)+sqrt(x(:,3))+x(:,4));
            f2 = F*L/E*(2./x(:,1)+2*sqrt(2)./x(:,2)-2*sqrt(2)./x(:,3)+2./x(:,4));
            F = [f1 f2]; %#ok<NASGU>
        end

        function F = re24(obj,x)
            x1=x(:,1); x2=x(:,2); E=700000;
            f1=x1+120*x2; sigmaB=4500./(x1.*x2); tau=1800./x2;
            delta=562000./(E*x1.*x2.^2); sigmaK=E*x1.^2/100;
            g1=1-sigmaB/700; g2=1-tau/450; g3=1-delta/1.5; g4=1-sigmaB./sigmaK;
            F=[f1 obj.violation(g1,g2,g3,g4)];
        end

        function F = re31(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3); f1=x1.*sqrt(16+x3.^2)+x2.*sqrt(1+x3.^2);
            f2=safeDivide(20*sqrt(16+x3.^2),x3.*x1);
            g1=0.1-f1; g2=100000-f2; g3=100000-safeDivide(80*sqrt(1+x3.^2),x3.*x2);
            F=[f1 f2 obj.violation(g1,g2,g3)];
        end

        function F = re32(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);P=6000;L=14;E=30e6;G=12e6;
            f1=1.10471*x1.^2.*x2+0.04811*x3.*x4.*(14+x2); f2=4*P*L^3./(E*x4.*x3.^3);
            M=P*(L+x2/2); R=sqrt(x2.^2/4+((x1+x3)/2).^2); J=2*sqrt(2)*x1.*x2.*(x2.^2/12+((x1+x3)/2).^2);
            tdd=M.*R./J; td=P./(sqrt(2)*x1.*x2); tau=sqrt(td.^2+(2*td.*tdd.*x2)./(2*R)+tdd.^2); sigma=6*P*L./(x4.*x3.^2);
            pc=(4.013*E.*sqrt(x3.^2.*x4.^6/36)/L^2).*(1-(x3/(2*L))*sqrt(E/(4*G)));
            F=[f1 f2 obj.violation(13600-tau,30000-sigma,x4-x1,pc-P)];
        end

        function F = re34(obj,x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);x5=x(:,5);
            f1=1640.2823+2.3573285*x1+2.3220035*x2+4.5688768*x3+7.7213633*x4+4.4559504*x5;
            f2=6.5856+1.15*x1-1.0427*x2+0.9738*x3+0.8364*x4-0.3695*x1.*x4+0.0861*x1.*x5+0.3628*x2.*x4-0.1106*x1.^2-0.3437*x3.^2+0.1764*x4.^2;
            f3=-0.0551+0.0181*x1+0.1024*x2+0.0421*x3-0.0073*x1.*x2+0.024*x2.*x3-0.0118*x2.*x4-0.0204*x3.*x4-0.008*x3.*x5-0.0241*x2.^2+0.0109*x4.^2;
            F=[f1 f2 f3];
        end

        function F = re35(obj,x)
            x1=x(:,1);x2=x(:,2);x3=round(x(:,3));x4=x(:,4);x5=x(:,5);x6=x(:,6);x7=x(:,7);
            f1=0.7854*x1.*x2.^2.*(10*x3.^2/3+14.933*x3-43.0934)-1.508*x1.*(x6.^2+x7.^2)+7.477*(x6.^3+x7.^3)+0.7854*(x4.*x6.^2+x5.*x7.^2);
            f2=sqrt((745*x4./(x2.*x3)).^2+1.69e7)./(0.1*x6.^3);
            g1=-1./(x1.*x2.^2.*x3)+1/27; g2=-1./(x1.*x2.^2.*x3.^2)+1/397.5; g3=-x4.^3./(x2.*x3.*x6.^6)+1/1.93; g4=-x5.^3./(x2.*x3.*x7.^6)+1/1.93;
            g5=-x2.*x3+40;g6=-x1./x2+12;g7=-5+x1./x2;g8=-1.9+x4-1.5*x6;g9=-1.9+x5-1.1*x7;g10=-f2+1300;
            g11=-sqrt((745*x5./(x2.*x3)).^2+1.575e8)./(0.1*x7.^3)+1100;
            F=[f1 f2 obj.violation(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11)];
        end

        function F = re37(obj,x)
            a=x(:,1);h=x(:,2);o=x(:,3);t=x(:,4);
            f1=0.692+0.477*a-0.687*h-0.080*o-0.065*t-0.167*a.^2-0.0129*h.*a+0.0796*h.^2-0.0634*o.*a-0.0257*o.*h+0.0877*o.^2-0.0521*t.*a+0.00156*t.*h+0.00198*t.*o+0.0184*t.^2;
            f2=0.153-0.322*a+0.396*h+0.424*o+0.0226*t+0.175*a.^2+0.0185*h.*a-0.0701*h.^2-0.251*o.*a+0.179*o.*h+0.015*o.^2+0.0134*t.*a+0.0296*t.*h+0.0752*t.*o+0.0192*t.^2;
            f3=0.370-0.205*a+0.0307*h+0.108*o+1.019*t-0.135*a.^2+0.0141*h.*a+0.0998*h.^2+0.208*o.*a-0.0301*o.*h-0.226*o.^2+0.353*t.*a-0.0497*t.*o-0.423*t.^2+0.202*h.*a.^2-0.281*o.*a.^2-0.342*h.^2.*a-0.245*h.^2.*o+0.281*o.^2.*h-0.184*t.^2.*a-0.281*h.*a.*o;
            F=[f1 f2 f3];
        end

    end
end
function y = safeDivide(a,b)
    % PlatEMO may provide a row or column batch depending on the caller.
    % Normalize both operands so logical indexing always has matching shape.
    a = a(:); b = b(:);
    y=zeros(size(a)); mask=(b~=0); y(mask)=a(mask)./b(mask);
end
