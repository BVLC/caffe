function score = GetPascalOverlap(bb, bbgt)
% Directly copied from Pascal code
%
% Gets the overlap measure according to Pascal
%
% bb:           Bounding Box
% bbgt:         Ground truth bounding box
%
% score:        Score between 0 and 1. 1 is complete overlap.

score = 0;

% intersection bbox
bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
iw=bi(3)-bi(1)+1;
ih=bi(4)-bi(2)+1;
if iw>0 & ih>0 % intersection should be non-zero               
    % compute overlap as area of intersection / area of union
    ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
       (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
       iw*ih;
    score=iw*ih/ua;
end