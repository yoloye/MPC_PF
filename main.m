% ====================================================================================================================
%                                          Copyright 2019 by Mohamed W. Mehrez & Wenrui Ye
%                                                       All rights reserved. 
% ====================================================================================================================

addpath('C:\Program Files\casadi')
import casadi.*
clc; 
close all;

% ====================================================================================================================
%                                                       Set up the system
% ====================================================================================================================
Time_period = 25; % Total simulate time
T = 0.05; % Sample time(s)
N = 30; % Prediction Horizon	
t_vector = (0:T:Time_period);

% Vehicle's information
C_f = 127000;
C_r = 130000;
I_z = 4600;
lf = 1.421;
lr = 1.434;
m = 2270;

% Obstacle information
num_obstacles = 1;
x_x_1 = 100; %initial longitudial position of first obstacle;
x_y_1 = 1.5; %initial lateral position of first obstacle;
V_1 = 15; %Velocity of first obstacle;

x0 = [0;25;1.5;0;0;0];% initialize states, based on number of states
xs = [500;25;6; 0;0;0]; %Reference state


% ====================================================================================================================
%                                            Set up the Control problem with casadi
% ====================================================================================================================
% Set up states of system model
X_s = SX.sym('X'); u_s = SX.sym('u_s');
Y = SX.sym('Y'); v = SX.sym('v');
theta = SX.sym('theta'); r = SX.sym('r');
states = [X_s; u_s; Y; v; theta;r];
n_states = length(states);

% Set up control(input) of system model
F = SX.sym('F');
delta = SX.sym('delta');
controls = [F; delta];
n_controls = length(controls);

F_yf = C_f*(delta-(v+lf*r)/u_s);
F_yr = C_r*(-(v-lr*r)/u_s);

rhs = [u_s*cos(theta)-v*sin(theta); F/m+v*r; v*cos(theta)+u_s*sin(theta); (F_yf+F_yr)/m-u_s*r; r; (lf*F_yf-lr*F_yr)/I_z];


% ====================================================================================================================
%                                                 Transfer OCP to NLP
% ====================================================================================================================
f = Function('f', {states, controls}, {rhs});
% Define a function "f", it will take states and controls as variable and return rhs

U = SX.sym('U', n_controls, N); %Store information of Decision Variables(controls)

p = SX.sym('p', n_states + n_states + n_controls + num_obstacles);
% n_states Current State
% n_states Reference State
% n_controls For constraint on Control related to previous state
% num_obstacles number of obstacles
% Parameters 'p', this part of this matrix(Length(n_states)) store 
% information of initial state of x(x0), and second part of it
% contains information of the reference state.

X = SX.sym('x', n_states,(N+1));
% A vector that contains states over the pridiction horizon period
% and the N+1 means that it contains the intial condition from last
% control period.

obj = 0 ; %objective function, will be continous update
g = []; %constrain vector


% ====================================================================================================================
%                                                  Weighting matrix                    
% ====================================================================================================================
%Weighting matrix of states
Q = zeros(n_states, n_states); 
Q(1,1) =0.1; Q(2,2) = 0.2; Q(3,3) = 5; 
Q(4,4) =0; Q(5,5) = 0; Q(6,6) = 0;

%Weighting matrix of controls
R = zeros(n_controls, n_controls); 
R(1,1) = 2e-8; R(2,2) = 500;

%Weighting matrix of increment within prediction horizon
S = zeros(n_controls, n_controls); 
S(1,1) = 5e-9; S(2,2) = 500;

%Weighting matrix of increment related to last control input
M = zeros(n_controls, n_controls);
M(1,1) = 5e-9; M(2,2) = 500;


% ====================================================================================================================
%                                                  Set up NLP solver                    
% ====================================================================================================================
st = X(:, 1); 
steer = U(2,1) ;
%X(:, 1) initial state, and this value will be updated in the following loop
g = [g; st - p(1:n_states)];
% initial condition constraints

% The following loop is aim to calculate the cost function and constrains
for i = 1:N
    zz = 0;
%   zz =  NonCrossable_PF(X(1, i), X(3, i),p(n_states * 2+1), x_y_1, X(2, i),X(4, i), V_1, 2, 1);
    if i == 1
        st = X(:, i);
        con = U(:, i);
        obj = obj + zz + (st - p((n_states + 1):(n_states * 2)))' * Q *...
            (st - p((n_states + 1):(n_states * 2))) + con' * R * con +...
            (U(:, i)-U(:, i-1))'*S*(U(:, i)-U(:, i-1))+...
            (p((n_states*2+1):(n_states*2+n_controls)))'*M*(p((n_states*2+1):(n_states*2+n_controls)));    
        st_next = X(:, i + 1);
        f_value = f(st, con); % this is define in previous step, it will return rhs
        st_next_predict = st+ (T*f_value); % the state of next time step
        g = [g; st_next - st_next_predict];
    else     
        st = X(:, i);
        con = U(:, i);            
        obj = obj + zz + (st - p((n_states + 1):(n_states * 2)))' * Q *...
            (st - p((n_states + 1):(n_states * 2))) + con' * R * con +...
            (U(:, i)-U(:, i-1))'*S*(U(:, i)-U(:, i-1));
        st_next = X(:, i + 1);
        f_value = f(st, con); 
        st_next_predict = st+ (T*f_value);
        g = [g; st_next - st_next_predict];
        % paraller compute the constrains, it equals state_next - state_next_predict
    end
end

for i = 1:N-1
    g = [g ; U(2, i+1) - U(2,i)];
end

for i = 1: N
    g = [g; X(2, i + 1) - X(2, i)];
end

% Form the NLP 
OPT_variables = [reshape(X, n_states * (N+1),1); reshape(U, n_controls * N,1)];
% reshape all predict states and controls variable to a one column vector
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', p);
opts = struct;
opts.ipopt.max_iter = 3000;
% the algorithm will terminate with an error after max_iter times iterations
opts.ipopt.print_level = 0;
% The larger the value, the more detail in output. [0 12]
opts.print_time = 0;
% A boolean value. print informatin of execution time
opts.ipopt.acceptable_tol = 1e-15;
% The convergence tolerance
opts.ipopt.acceptable_obj_change_tol = 1e-6;
%Stop criterion
solver = nlpsol('solver', 'ipopt', nlp_prob, opts);


% ====================================================================================================================
%                                              Set up the constraints
% ====================================================================================================================
%Following constraint should be adjust based on specific problem
args = struct;

args.lbg(1: n_states * (N+1)) = 0;
args.ubg(1: n_states * (N+1)) = 0;

% Increment of steering angle on front wheel
args.lbg( n_states * (N+1)+1:n_states * (N+1)+N-1) = -5e-9;
args.ubg( n_states * (N+1)+1:n_states * (N+1)+N-1) = 5e-9;

% Increment of longituidal speed
args.lbg( n_states * (N+1)+N :n_states * (N+1)+2*N-1 ) = -0.1;
args.ubg( n_states * (N+1)+N : n_states * (N+1)+2*N-1) = 0.1;

%Longituidal position 
args.lbx(1: n_states: n_states * (N+1), 1) = -inf;
args.ubx(1: n_states: n_states * (N+1), 1) = inf;

%Longitudinal Velocity u <100km/h
args.lbx(2: n_states: n_states * (N+1), 1) = 0;
args.ubx(2: n_states: n_states * (N+1), 1) = 27.8;

%Lateral position
args.lbx(3: n_states: n_states * (N+1), 1) = 1;
args.ubx(3: n_states: n_states * (N+1), 1) = 6; % consider width of vehicle body

%Lateral speed
args.lbx(4: n_states: n_states * (N+1), 1) = -3;
args.ubx(4: n_states: n_states * (N+1), 1) = 3;

% Yaw angle 
args.lbx(5: n_states: n_states * (N+1), 1) = -pi/12;
args.ubx(5: n_states: n_states * (N+1), 1) = pi/12 ;

%Yaw rate
args.lbx(6: n_states: n_states * (N+1), 1) = -0.5;
args.ubx(6: n_states: n_states * (N+1), 1) = 0.5;

%Constrains of control variable u(steering angle)
%Longitudial tire force
args.lbx(n_states*(N+1)+1: 1: n_states * (N+1) +n_controls * N,1) = -5000;
args.ubx(n_states*(N+1)+1: 1: n_states * (N+1) +n_controls * N,1) = 5000;

%Steering angle on front wheel
args.lbx(n_states*(N+1)+2: 2: n_states * (N+1) +n_controls * N,1) = -pi/2;
args.ubx(n_states*(N+1)+2: 2: n_states * (N+1) +n_controls * N,1) = pi/2;


% ====================================================================================================================
%                                              Set up the simulation loop
% ====================================================================================================================
t0 = 0;
x_p_1 = []; % information of obstacle 1
u_prev = [0, 0];
xx(:, 1) = x0;
t(1) = t0;
u0 = zeros(N, n_controls);
X0 = repmat(x0, 1, N+1)';
% refer as repeat matrix, a n+1 * 1 cell matrix , and each cell contains
% the matrix x0, the cell to mat

sim_time = Time_period;
mpciter = 0;
xx1 = [];
u_cl = [];
main_loop = tic; % timer on
i = 1;

while(norm((x0 - xs),2) > 1e-2 && mpciter < sim_time/T)
%   while mpciter < sim_time/T)
%   this condition is check the error
    args.p = [x0 ; xs;u_prev(1,1); u_prev(1,2); x_x_1];
    args.x0 = [reshape(X0',n_states * (N+1),1); reshape(u0', n_controls * N,1)];
    sol = solver('x0',args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
    u = reshape(full(sol.x(n_states * (N+1)+1:end))', n_controls , N)';
    xx1(:,1:n_states, mpciter+1) = reshape(full(sol.x(1:n_states * (N+1)))', n_states, N+1)';
%     this is a 3-D matrix, get controls from solution
%     if (u(1,2) - u_cl(i,2)) > (0.005*T)
%         u(1,2) = u_cl(i,2) + (0.005*T);
%     elseif (u(1,2) - u_cl(i,2)) < -(0.01*T)
%         u(1,2) = u_cl(i,2) - (0.01*T);
%     end
    u_cl = [u_cl; u(1,:)];
    [m,n] = size(u_cl);
    u_prev(1,1) =u_cl(m,1);
    u_prev(1,2) =u_cl(m,2);
%   This matrix save the information of control strategy
%   This is the most important information, use this to plot
    t(mpciter+1) = t0;
    x_p_1 = [x_p_1 ; x_x_1];
    [t0, x0, u0, x_x_1] = shift(T, t0, x0, u, f,x_x_1,V_1);
    xx(:, mpciter + 2) = x0;
%   This matrix save the information of state
%   This is the most important information, use this to plot
    X0 = reshape(full(sol.x(1:n_states*(N+1)))',n_states,N+1)'; % get solution TRAJECTORY
    X0 = [X0(2:end,:); X0(end,:)];
    mpciter = mpciter + 1;
    i = i +1 ;
end

main_loop_time = toc(main_loop) % display simulation time
ss_error = norm((x0-xs),2)
average_mpc_time = main_loop_time/(mpciter+1);


% ====================================================================================================================
%                                                           Plot
% ====================================================================================================================
%Figure 1 is figure of all six states
figure(1)
title('six states','FontSize', 12)
for i = 1:6
subplot(n_states,1,i)
plot(xx(i,:),'LineWidth',2)
hold on
end

%Figure 2 is figure of all control input
figure(2)
control_t_vector = (0:T:Time_period-T);
subplot(2,1,1)
plot(control_t_vector, 57.3*u_cl(:,2),'LineWidth',2)
title('steering input','FontSize', 12)
xlabel('Time','FontSize', 12)
ylabel('Steering angle on wheel(deg)','FontSize', 12)
subplot(2,1,2)
plot(control_t_vector, u_cl(:,1),'LineWidth',2)
title('Force','FontSize', 12)
xlabel('Time','FontSize', 12)
ylabel('Force (N)','FontSize', 12)

%Figure 3 is figure of the Vehicle Path vs. Time
figure(3)
x_time = 0:T:Time_period;
plot(x_time,xx(3,:),'LineWidth',2);
ylim([0,7.5])
title('Vehicle Lateral Position vs.Time','FontSize', 12)
xlabel('Time','FontSize', 12)
ylabel('Lateral Position','FontSize', 12)

%Figure 4 is figure of the Longitudial Position vs. Lateral Position 
figure(4)
size_x = size(xx);
y = xx(3,:);
x_time = 0:T:Time_period;
x = xx(1,:);
plot(x,y,'LineWidth',2);
ylim([0,7.5])
pbaspect([10 1 1])
title('Vehicle Position','FontSize', 12)
xlabel('Longitudinal Position','FontSize', 12)
ylabel('Lateral Position','FontSize', 12)


%Figure 5 is figure of the Yaw angle vs. Time
figure(5)
plot(t_vector,xx(5,:),'LineWidth',2)
title('Yaw angle vs. Time','FontSize', 12)
xlabel('Time','FontSize', 12)
ylabel('Yaw angle(rad)','FontSize', 12)

%Figure 6 is figure of Lateral Acceleration & steering angle & Yaw rate
figure(6)
%Follwing code is used to calculate lateral accleration
V_x = xx(2,:);
V_y = xx(4,:);
beta = atan(V_y./V_x);
delta_beta=[0];
accy =[0];
for i = 1: (Time_period/T)
    delta_beta(i+1) = (beta(i+1)-beta(i))/T;
    V = sqrt(xx(2,i)^2 +xx(4,i)^2);
    acc_y(i) = (delta_beta(i)-xx(6,i))*V;
end
yyaxis right
plot(t_vector(1:(length(acc_y))), acc_y,'b','LineWidth',2)
ylabel('m/s^2','FontSize', 12)
hold on
yyaxis left
plot( 57.3*u_cl(:,2),'r','LineWidth',2)
hold on
plot(xx(6,:)*57.3,'k','LineWidth',2)
xlim([0 25])
title('Lateral Acceleration & steering angle & Yaw rate','FontSize', 12)
xlabel('Time','FontSize', 12)
ylabel('degree & degree/s','FontSize', 12)
legend('Steering angle', 'Yaw rate','Lateral Acceleration')

%Figure 7 is figure of Yaw angle & Yaw rate & Path
figure(7)
yyaxis right
plot(t_vector,xx(6,:)*57.3,'LineWidth',2)
hold on
plot(t_vector,xx(5,:)*57.3,'k','LineWidth',2)
% ylim([-4 4])
hold on
ylabel('degree & degree/s ','FontSize', 12)
yyaxis left
plot(t_vector,xx(3,:),'LineWidth',2)
% ylim([-3 6])
legend('path','Y_Velocity', 'Yaw rate')
xlim([0 25])
legend( 'path', 'Yaw rate', 'Yaw angle')
xlabel('Time','FontSize', 12)
ylabel('m','FontSize', 12)
title('Yaw angle & Yaw rate & Path')

%Figure 8 is figure of Longitudinal Speed
figure(8)
plot(t_vector, xx(2,:),'LineWidth',2)
xlabel('Time','FontSize', 12)
ylabel('Longitudinal Velocity m/s','FontSize', 12)
title('Longitudinal Speed')

%Figure 9 is a simple animation of vehivle and obstacle's path
myVideo = VideoWriter('myVideoFile'); %open video file
myVideo.FrameRate = 50;  %can adjust this
open(myVideo)
figure(9)
xxx =x_p_1';
x_time = 1.5*ones(1, 500);
for ind=1:length(xx)-1
    plot(xx(1,1:ind),xx(3,1:ind),'r','LineWidth',2)
    hold on
    plot(xxx(1,1:ind), x_time(1,1:ind),'b','LineWidth',2)
    xlim([0,500])
    ylim([0,7.5])
    pbaspect([10 1 1])
    title('Vehicle Position','FontSize', 12)
    xlabel('Longitudinal Position','FontSize', 12)
    ylabel('Lateral Position','FontSize', 12)
    drawnow
    pause(0.001)
    frame = getframe(gcf); %get frame
    writeVideo(myVideo, frame);
    % axis([min(t) max(t) min(x) max(x)])
end
close(myVideo)








