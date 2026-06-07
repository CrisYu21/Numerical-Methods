function result = matrix_toolkit(operation, A, B, p)
    % FINALIZED MASTER ROUTER FOR ALL 8 REQUIRED MATRIX OPERATIONS
    % Inputs:
    %   operation : String selecting the math operation
    %   A, B      : Input matrices passed from the user interface
    %   p         : Scalar integer used exclusively for 'Power of Matrix'
    % Output:
    %   result    : The calculated matrix or scalar solution

    switch operation
        case 'Matrix Addition'
            if ~isequal(size(A), size(B))
                error('Matrices must be of identical dimensions for addition.');
            end
            result = A + B;

        case 'Matrix Multiplication'
            if size(A, 2) ~= size(B, 1)
                error('Inner matrix dimensions must agree (Columns of A must equal Rows of B).');
            end
            result = A * B;

        case 'Adjoint'
            if size(A, 1) ~= size(A, 2)
                error('Adjoint calculation requires a square matrix.');
            end
            if det(A) == 0
                error('Matrix is singular (det=0). Cannot compute Adjoint via inverse method.');
            end
            % Mathematical formula: adj(A) = det(A) * inv(A)
            % Use backslash to avoid explicit inv(A) for numeric stability.
            result = det(A) * (A \ eye(size(A)));

        case 'Inverse Matrix'
            if size(A, 1) ~= size(A, 2)
                error('Matrix inversion requires a square matrix.');
            end
            if det(A) == 0
                error('Matrix is singular and cannot be inverted (Determinant = 0).');
            end
            result = A \ eye(size(A));

        case 'Determinants'
            if size(A, 1) ~= size(A, 2)
                error('Determinant calculation requires a square matrix.');
            end
            result = det(A);

        case 'Power of Matrix'
            % SMART FIX: If the app only sends 3 inputs, the exponent is sitting 
            % in the 'B' variable. We assign it to 'p' so the math doesn't break!
            if nargin < 4
                p = B;
            end

            if size(A, 1) ~= size(A, 2)
                error('Raising a matrix to a power requires a square matrix.');
            end
            
            if isempty(p)
                error('Please provide an integer power exponent (p).');
            end
            
            result = A^p;
        
        case 'Equations'
            % Solves AX = B system of linear equations
            if size(A, 1) ~= size(A, 2)
                error('Coefficient Matrix A must be square.');
            end
            if size(A, 1) ~= size(B, 1)
                error('Row dimensions of Constant Matrix B must match Matrix A.');
            end
            result = A \ B; % Uses high-performance Gaussian elimination solver

        case 'Transpose of Matrix'
            result = A.';

        otherwise
            error('Matrix operation not recognized.');
    end
end