% ====================================================================================================================
%                                            Copyright 2019 by Mohamed W. Mehrez & Wenrui Ye
%                                                       All rights reserved. 
% ====================================================================================================================
% ====================================================================================================================
% [zz] = RoadBoundary_PF(yy)   
% RoadBoundary_PF is a function that takes value of road 2-D dimension
% Effect: Takes value of road 2-D dimension, Calculate PF of road based on road 2-D dimension 
%           and return value of PF. The road only has two lanes.
% Variables: xlb(Road length lower boundary) and xub(Road length upper boundary)
%            !!!!The width of road is 3.75!!!!
% Returns: xx(The width of road(matrix)), yy(The length of road(matrix)) and
%          zz(The potential field generated by road(matrix)).
% Example: [zz] = RoadBoundary_PF(5)
% Use surf(xx,yy,zz) can draw teh surface graph of potential field.
% ====================================================================================================================
function [zz] = RoadBoundary_PF(yy)   
    zz = 0.5*(0.0128*yy^4-0.192*yy^3+0.9505*yy^2-1.724*yy+1);
end