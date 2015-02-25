function intersection = BoxIntersection(a, b)
% intersection = BoxIntersection(a, b)
%
% Creates the intersection of two bounding boxes. Returns minus ones if
% there is no intersection
%
% a:            Input bonding box "a"
% b:            Input bounding box "b"
%
% intersection: Intersection of box a and b
%
%     Jasper Uijlings - 2013

intersection = [max(a(:,1),b(:,1)) max(a(:,2),b(:,2)) ...
                min(a(:,3),b(:,3)) min(a(:,4),b(:,4))];
                
[numRows numColumns] = BoxSize(intersection);

% There is no intersection box
negIds = numRows < 1 | numColumns < 1;
intersection(negIds,:) = -1;


