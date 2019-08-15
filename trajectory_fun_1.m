% ====================================================================================================================
%                                          Copyright 2019 by Mohamed W. Mehrez & Wenrui Ye
%                                                       All rights reserved. 
% ====================================================================================================================
% ====================================================================================================================
% [long_pos,lateral_pos] = trajectory_fun_1(Time_step, Time_period, road_width, v_x)
% Effect: The function takes the time information, road information, speed of vehicle 
%         and gives you a series of point on a sine wave
% Variables: Time_step: Simulation time step, Time_period: Total simulation time
%        road_width: width of the road or lane depends on how you define,
%        v_x: longitudial speed of the ego vehicle
% Return: long_pos: A matrix of Longitudial position at each time step,
%         lateral_pos: A matrix of Lateral position at each time step
%Example: [long_pos,lateral_pos] = trajectory_fun_1(0.05, 30, 7, 25)
% ====================================================================================================================
function [long_pos,lateral_pos] = trajectory_fun_1(Time_step, Time_period, road_width, v_x)
    a = 0:Time_step:Time_period;
    long_pos = a * v_x;
    lateral_pos = (road_width-1)/2+1+(road_width-1)/2*sin(0.015*long_pos);
%     plot(long_pos,lateral_pos)
%     ylim([0,7])
end
