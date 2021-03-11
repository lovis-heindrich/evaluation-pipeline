classification_res = readtable("Classification_raw_results.csv");

disp('Classification res')

classification_res{:, "Level"} = erase(classification_res{:, "Level"}, 'Level ');
classification_res.Level = str2double(classification_res.Level);
% ANOVA for acc
[p, tmp, statsOut] = anovan(classification_res.Accuracy, {classification_res.Classifier, classification_res.Transformation, classification_res.Level}, 'varnames', {'classifier', 'transformation', 'level'}, 'continuous', [3]);
[c, m, h] =multcompare(statsOut, 'Dimension', [2]);
disp('Accuracy - p-values for (classifier, transformation, level)')
disp(p)
set(h,'SelectionHighlight','off')
saveas(h, "classification_acc_transformation.png")

[p, tmp, statsOut] = anovan(classification_res.ROC, {classification_res.Classifier, classification_res.Transformation, classification_res.Level}, 'varnames', {'classifier', 'transformation', 'level'}, 'continuous', [3]);
disp('ROC - p-values for (classifier, transformation, level)')
disp(p)

[p, tmp, statsOut] = anovan(classification_res.Brier, {classification_res.Classifier, classification_res.Transformation, classification_res.Level}, 'varnames', {'classifier', 'transformation', 'level'}, 'continuous', [3]);
disp('Brier - p-values for (classifier, transformation, level)')
disp(p)