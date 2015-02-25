function union = BoxUnion(a, b)
% union = BoxUnion(a, b)
%
% Creates the union box of two bounding boxes. 
%
% a:            Input bonding box "a"
% b:            Input bounding box "b"
%
% union: Intersection of box a and b
%
%     Jasper Uijlings - 2013

union = [min(a(:,1),b(:,1)) min(a(:,2),b(:,2)) ...
         max(a(:,3),b(:,3)) max(a(:,4),b(:,4))];
