function res = voc_eval(path, comp_id, test_set, output_dir)

VOCopts = get_voc_opts(path);
VOCopts.testset = test_set;

for i = 1:length(VOCopts.classes)
  cls = VOCopts.classes{i};
  res(i) = voc_eval_cls(cls, VOCopts, comp_id, output_dir);
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results:\n');
aps = [res(:).ap]';
fprintf('%.1f\n', aps * 100);
fprintf('%.1f\n', mean(aps) * 100);
fprintf('~~~~~~~~~~~~~~~~~~~~\n');

function res = voc_eval_cls(cls, VOCopts, comp_id, output_dir)

test_set = VOCopts.testset;
year = VOCopts.dataset(4:end);

addpath(fullfile(VOCopts.datadir, 'VOCcode'));

res_fn = sprintf(VOCopts.detrespath, comp_id, cls);

recall = [];
prec = [];
ap = 0;
ap_auc = 0;

do_eval = (str2num(year) <= 2007) | ~strcmp(test_set, 'test');
if do_eval
  % Bug in VOCevaldet requires that tic has been called first
  tic;
  [recall, prec, ap] = VOCevaldet(VOCopts, comp_id, cls, true);
  ap_auc = xVOCap(recall, prec);

  % force plot limits
  ylim([0 1]);
  xlim([0 1]);

  print(gcf, '-djpeg', '-r0', ...
        [output_dir '/' cls '_pr.jpg']);
end
fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

res.recall = recall;
res.prec = prec;
res.ap = ap;
res.ap_auc = ap_auc;

save([output_dir '/' cls '_pr.mat'], ...
     'res', 'recall', 'prec', 'ap', 'ap_auc');

rmpath(fullfile(VOCopts.datadir, 'VOCcode'));
