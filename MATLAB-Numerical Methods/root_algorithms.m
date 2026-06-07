function [root, T] = root_algorithms(method, eq_str, a, b, tol, maxIter)
    % 1. Convert string to symbolic and then to a function handle
    % The sanitization (2x -> 2*x) is already handled by the callback
    syms x;
    try
        f_sym = str2sym(eq_str);
        
        % Validate that only 'x' is used
        if ~isempty(setdiff(symvar(f_sym), x))
            error('Equation must only contain variable "x".');
        end
        
        % Convert to function handle, explicitly setting 'x' as the variable
        f = matlabFunction(f_sym, 'vars', x);
        
    catch ME
        error('Error processing equation: %s', ME.message);
    end

    % 2. Dispatch to the appropriate engine
    switch method
        case 'Increment', [root, T] = run_increment(f, a, b, tol, maxIter);
        case 'Bisection', [root, T] = run_bisection(f, a, b, tol, maxIter);
        case 'Regula-Falsi', [root, T] = run_regula_falsi(f, a, b, tol, maxIter);
        case 'Newton-Raphson'
            % Create derivative function handle explicitly
            df = matlabFunction(diff(f_sym, x), 'vars', x); 
            [root, T] = run_newton(f, df, a, tol, maxIter);
        case 'Secant', [root, T] = run_secant(f, a, b, tol, maxIter);
        otherwise
            error('Unknown method selected.');
    end
end

%% --- ENGINES ---

function [root, T] = run_increment(f, a, b, tol, maxIter)
    % Initialize root so it ALWAYS has a value
    root = NaN; 
    % For Increment: a=x_start, b=dx (step size)
    x = a; dx = b; count = 0; 
    Iter = zeros(maxIter,1); XL = zeros(maxIter,1); XR = zeros(maxIter,1); 
    XU = zeros(maxIter,1); fXL = zeros(maxIter,1); fXR = zeros(maxIter,1); 
    Ea = zeros(maxIter,1); Prod = zeros(maxIter,1); Rem = cell(maxIter,1);
    
    for i=1:maxIter
        count = i; x_next = x + dx;
        Iter(i)=i; XL(i)=x; XR(i)=x_next; XU(i)=a + dx*maxIter; 
        fXL(i)=f(x); fXR(i)=f(x_next); Ea(i)=abs(dx); Prod(i)=f(x)*f(x_next);
        
        if sign(f(x)) ~= sign(f(x_next))
            Rem{i}='Sign change!'; root=[x, x_next]; break; 
        elseif abs(f(x)) < tol
            Rem{i}='Converged'; root=x; break;
        else
            Rem{i}='Next';
        end
        x = x_next;
    end
    T = table(Iter(1:count), XL(1:count), XR(1:count), XU(1:count), ...
              fXL(1:count), fXR(1:count), Ea(1:count), Prod(1:count), Rem(1:count), ...
              'VariableNames', {'Iteration', 'XL', 'XR', 'XU', 'f_XL', 'f_XR', 'Ea', 'Product', 'Remarks'});
end
function [root, T] = run_bisection(f, a, b, tol, maxIter)
    Iter=zeros(maxIter,1); XL=zeros(maxIter,1); XR=zeros(maxIter,1); XU=zeros(maxIter,1); fXL=zeros(maxIter,1); fXR=zeros(maxIter,1); Ea=zeros(maxIter,1); Prod=zeros(maxIter,1); Rem=cell(maxIter,1);
    count=0; root=NaN; c_old=a;
    for i=1:maxIter
        count=i; c = (a+b)/2; Iter(i)=i; XL(i)=a; XU(i)=b; XR(i)=c; fXL(i)=f(a); fXR(i)=f(c); Ea(i)=abs(c-c_old); Prod(i)=f(a)*f(c);
        if abs(f(c))<tol || Ea(i)<tol, root=c; Rem{i}='Converged'; break; else, Rem{i}='Next'; end
        if Prod(i)<0, b=c; else, a=c; end
        c_old=c;
    end
    T = table(Iter(1:count), XL(1:count), XR(1:count), XU(1:count), fXL(1:count), fXR(1:count), Ea(1:count), Prod(1:count), Rem(1:count), 'VariableNames', {'Iteration', 'XL', 'XR', 'XU', 'f_XL', 'f_XR', 'Ea', 'Product', 'Remarks'});
end

function [root, T] = run_regula_falsi(f, a, b, tol, maxIter)
    Iter=zeros(maxIter,1); XL=zeros(maxIter,1); XR=zeros(maxIter,1); XU=zeros(maxIter,1); fXL=zeros(maxIter,1); fXR=zeros(maxIter,1); Ea=zeros(maxIter,1); Prod=zeros(maxIter,1); Rem=cell(maxIter,1);
    count=0; root=NaN; c_old=a;
    for i=1:maxIter
        count=i; c = b - (f(b)*(a-b))/(f(a)-f(b)); Iter(i)=i; XL(i)=a; XU(i)=b; XR(i)=c; fXL(i)=f(a); fXR(i)=f(c); Ea(i)=abs(c-c_old); Prod(i)=f(a)*f(c);
        if abs(f(c))<tol || Ea(i)<tol, root=c; Rem{i}='Converged'; break; else, Rem{i}='Next'; end
        if Prod(i)<0, b=c; else, a=c; end
        c_old=c;
    end
    T = table(Iter(1:count), XL(1:count), XR(1:count), XU(1:count), fXL(1:count), fXR(1:count), Ea(1:count), Prod(1:count), Rem(1:count), 'VariableNames', {'Iteration', 'XL', 'XR', 'XU', 'f_XL', 'f_XR', 'Ea', 'Product', 'Remarks'});
end

function [root, T] = run_newton(f, df, x0, tol, maxIter)
    root = NaN; Iter=zeros(maxIter,1); XL=NaN(maxIter,1); XR=zeros(maxIter,1); 
    XU=zeros(maxIter,1); fXL=NaN(maxIter,1); fXR=zeros(maxIter,1); 
    Ea=zeros(maxIter,1); Prod=NaN(maxIter,1); Rem=cell(maxIter,1);
    
    count=0; x=x0;
    for i=1:maxIter
        count=i; x_next = x - f(x)/df(x);
        Iter(i)=i; XR(i)=x; XU(i)=x_next; fXR(i)=f(x_next); Ea(i)=abs(x_next-x);
        if abs(f(x_next))<tol || Ea(i)<tol, root=x_next; Rem{i}='Converged'; break; else, Rem{i}='Next'; end
        x=x_next;
    end
    T = table(Iter(1:count), XL(1:count), XR(1:count), XU(1:count), fXL(1:count), fXR(1:count), Ea(1:count), Prod(1:count), Rem(1:count), ...
        'VariableNames', {'Iteration', 'N_A1', 'Xi', 'Xi_plus_1', 'N_A2', 'f_Xi', 'f_Xi_plus_1', 'Product', 'Remarks'});
end

function [root, T] = run_secant(f, x0, x1, tol, maxIter)
    root = NaN; Iter=zeros(maxIter,1); XL=zeros(maxIter,1); XR=zeros(maxIter,1); 
    XU=zeros(maxIter,1); fXL=zeros(maxIter,1); fXR=zeros(maxIter,1); 
    Ea=zeros(maxIter,1); Prod=NaN(maxIter,1); Rem=cell(maxIter,1);
    
    count=0;
    for i=1:maxIter
        count=i; denom = f(x0)-f(x1);
        if abs(denom)<eps, error('Zero denom'); end
        x_next = x1 - (f(x1)*(x0-x1))/denom;
        Iter(i)=i; XL(i)=x0; XU(i)=x1; XR(i)=x_next; fXL(i)=f(x0); fXR(i)=f(x_next); Ea(i)=abs(x_next-x1);
        if abs(f(x_next))<tol || Ea(i)<tol, root=x_next; Rem{i}='Converged'; break; else, Rem{i}='Next'; end
        x0=x1; x1=x_next;
    end
    T = table(Iter(1:count), XL(1:count), XU(1:count), XR(1:count), fXL(1:count), fXR(1:count), Ea(1:count), Prod(1:count), Rem(1:count), ...
        'VariableNames', {'Iteration', 'Xi_minus_1', 'Xi', 'Xi_plus_1', 'f_Xi_minus_1', 'f_Xi', 'Ea', 'Product', 'Remarks'});
end