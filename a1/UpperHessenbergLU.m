function [L,U]=hlu(A)
  [n,n] = size(A); % A is nxn
  l = zeros(n-1,1);
  for row_index=1:n-1
    % multiplying factor
    l(row_index) = A(row_index+1, row_index)/A(row_index, row_index);
    % update row, i.e. M*A
    A(row_index+1, row_index:n) = A(row_index+1, row_index:n) ...
      - l(row_index)*A(row_index, row_index:n);
  end
  U = A;
  % L is just I + l (on the i=j+1 diagonal)
  L = diag(ones(n,1), 0) + diag(l, -1);
end