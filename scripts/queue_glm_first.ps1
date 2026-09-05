# Supersedes the queue in round_next.ps1 (21 -> 43 -> 48, all gpt).
#
# New order, per the user's instruction: the GLM-5.3 experiment runs FIRST once the
# in-flight L3:21 finishes, then the two remaining gpt tasks.
#
#   1. glm  level3:21   configs/experiments_l3_glm.yaml
#   2. gpt  level3:43   configs/experiments_l3.yaml
#   3. gpt  level3:48   configs/experiments_l3.yaml
#
# WHY level3:21 FOR THE GLM ARM: the gpt arm has just produced its best result of the
# project on 21 (9.78 ms tuned), and 21 has by far the most finished gpt runs to compare
# against. Running the new model on the SAME task makes it a model swap with a reference
# distribution; a different task would confound model with task.
#
# WaitForPid is the in-flight run's own python process, NOT the shell wrapping it: killing
# the owner of a `| Tee-Object` pipeline kills the child run (verified by experiment).
& "D:\Pyhon_projects\opop\v2\scripts\run_chain.ps1" `
  -WaitForPid 18516 `
  -Jobs @(
    @{ Task = "level3:21"; Config = "configs/experiments_l3_glm.yaml"; Label = "glm" },
    @{ Task = "level3:43"; Config = "configs/experiments_l3.yaml";     Label = "gpt" },
    @{ Task = "level3:48"; Config = "configs/experiments_l3.yaml";     Label = "gpt" }
  )
