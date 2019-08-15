% ====================================================================================================================
%                                          Copyright ? 2019 by Mohamed W. Mehrez & Wenrui Ye
%                                                       All rights reserved. 
% ====================================================================================================================
% ====================================================================================================================
% [t0, x0, u0, x_x_1] = shift(T, t0, x0, u, f, x_x_1, V_1) 
% shift is a function that calculate information of next time step
% Effect: Update t0, x0, u0, x_x_1 based on information of last time step
% Variables: T(sample time), t0(current time), x0(A matrix that contain information of all states),
%            u(control input based on whole prediction horizon, same length as prediction horizon),
%            f(The system function), x_x_1(position of obstacle), V_1(velocity of obstacle)
% Returns: t0(updated current time(next time step)), x0(A matrix that contain information of all states), 
%          u0(control input based on whole prediction horizon, same length as prediction horizon),
%          x_x_1(new position of obstacle)
% Example: [t0, x0, u0, x_x_1] = shift(T, t0, x0, u, f, x_x_1, V_1)
%          directly call this function in main function
% ====================================================================================================================
function [t0, x0, u0, x_x_1] = shift(T, t0, x0, u, f, x_x_1, V_1)    
    x_x_1 = x_x_1 +V_1*T;   
    st = x0;
    con = u(1,:)';
    f_value = f(st,con);
    st = st + (T * f_value);
    x0 = full(st);
    t0 = t0 + T;
    u0 = [u(2:size(u,1),:); u(size(u,1), :)];
    % Update u0, u from this step is the u0 of next step
    % This has to be change if there are more than one input
end