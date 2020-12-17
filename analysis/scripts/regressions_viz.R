library(tidyverse)
library(ggplot2)

setwd("E:/NFL/big_data_bowl/2021")

data = read_csv("data/created/model_coverage_wins.csv")

### Make individual logistic regressions for each variable of interest and find the point closest to 0.5 to highlight in the plot
## Yds. Past LOS
yds_past_los_fit = glm(target~ yds_thrown_past_los, data = data, family = binomial)

data$yds_past_los_preds = predict(yds_past_los_fit, type = 'response')
data$pred_minus_pt5 = abs(data$yds_past_los_preds-0.5)

fiddy_pt = data$yds_thrown_past_los[which.min(data$pred_minus_pt5)]
fiddy_pt_pred = data$yds_past_los_preds[which.min(data$pred_minus_pt5)]

ggplot(data, aes(x = yds_thrown_past_los, y = target)) + 
  geom_point(alpha=0.005) + 
  geom_line(aes(x=yds_thrown_past_los, y = yds_past_los_preds), size=1.5) +
  annotate("segment", x=fiddy_pt, xend= fiddy_pt, y = 0, yend=fiddy_pt_pred, colour='#4292c6', alpha = 0.5, size = 1.25) + 
  annotate("segment", x=fiddy_pt, xend= fiddy_pt+5, y = fiddy_pt_pred, yend=fiddy_pt_pred,
           colour='#4292c6', alpha=0.5, size = 1.25) + 
  annotate("label", x = fiddy_pt+5, y=fiddy_pt_pred, hjust = 0,
           label=paste0("At ~", round(fiddy_pt), " yds. downfield,\ncoverage wins about 50%,\non average.")) +
  labs(x = '\nYds. Beyond LOS', y = 'Coverage Win Rate\n', title = 'Defense wins a higher % of plays further downfield')

### Only including passes thrown 5+ yards downfield
data_downfield = data %>% filter(yds_thrown_past_los >= 5)

downfield_yds_past_los_fit = glm(target~ yds_thrown_past_los, data = data_downfield, family = binomial)

data_downfield$yds_past_los_preds = predict(downfield_yds_past_los_fit, type = 'response')
data_downfield$pred_minus_pt5 = abs(data_downfield$yds_past_los_preds-0.5)

fiddy_pt = data_downfield$yds_thrown_past_los[which.min(data_downfield$pred_minus_pt5)]
fiddy_pt_pred = data_downfield$yds_past_los_preds[which.min(data_downfield$pred_minus_pt5)]

ggplot(data_downfield, aes(x = yds_thrown_past_los, y = target)) + 
  geom_point(alpha=0.005) + 
  geom_line(aes(x=yds_thrown_past_los, y = yds_past_los_preds), size=1.5) +
  annotate("segment", x=fiddy_pt, xend= fiddy_pt, y = 0, yend=fiddy_pt_pred, colour='#4292c6', alpha = 0.5, size = 1.25) + 
  annotate("segment", x=fiddy_pt, xend= fiddy_pt+5, y = fiddy_pt_pred, yend=fiddy_pt_pred,
           colour='#4292c6', alpha=0.5, size = 1.25) + 
  annotate("label", x = fiddy_pt+5, y=fiddy_pt_pred, hjust = 0,
           label=paste0("At ~", round(fiddy_pt), " yds. downfield,\ncoverage wins about 50%,\non average.")) +
  labs(x = '\nYds. Beyond LOS', y = 'Coverage Win Rate\n', title = 'Defense wins a higher % of plays further downfield')


## Receiver distance to football
receiver_ball_dist_fit = glm(target~ receiver_distance_to_football, data = data_downfield, family = binomial)

data_downfield$receiver_ball_dist_preds = predict(receiver_ball_dist_fit, type = 'response')
data_downfield$rbd_pred_minus_pt5 = abs(data_downfield$receiver_ball_dist_preds-0.5)

fiddy_pt = data_downfield$receiver_distance_to_football[which.min(data_downfield$rbd_pred_minus_pt5)]
fiddy_pt_pred = data_downfield$receiver_ball_dist_preds[which.min(data_downfield$rbd_pred_minus_pt5)]

ggplot(data_downfield, aes(x = receiver_distance_to_football, y = target)) + 
  geom_point(alpha=0.005) + 
  geom_line(aes(x=receiver_distance_to_football, y = receiver_ball_dist_preds), size=1.5) +
  annotate("segment", x=fiddy_pt, xend= fiddy_pt, y = 0, yend=fiddy_pt_pred, colour='#4292c6', alpha = 0.5, size = 1.25) + 
  annotate("segment", x=fiddy_pt, xend= fiddy_pt+1.5, y = fiddy_pt_pred, yend=fiddy_pt_pred,
           colour='#4292c6', alpha=0.5, size = 1.25) + 
  annotate("label", x = fiddy_pt+1.5, y=fiddy_pt_pred, hjust = 0,
           label=paste0("When receivers are ~", round(fiddy_pt,1),
                        " yds. from the football,\ncoverage wins about 50%, on average.")) +
  xlim(0, 10) +
  labs(x = '\nRecevier Yds. From Football', y = 'Coverage Win Rate\n',
       title = 'Typical receiver catch radius is around 2.2 yds.')



## Receiver distance to defender
receiver_d_dist_fit = glm(target~ receiver_corner_dist_between, data = data_downfield, family = binomial)

data_downfield$receiver_d_dist_preds = predict(receiver_d_dist_fit, type = 'response')
data_downfield$rdd_pred_minus_pt5 = abs(data_downfield$receiver_d_dist_preds-0.5)

fiddy_pt = data_downfield$receiver_corner_dist_between[which.min(data_downfield$rdd_pred_minus_pt5)]
fiddy_pt_pred = data_downfield$receiver_d_dist_preds[which.min(data_downfield$rdd_pred_minus_pt5)]

ggplot(data_downfield, aes(x = receiver_distance_to_football, y = target)) + 
  geom_point(alpha=0.005) + 
  geom_line(aes(x=receiver_corner_dist_between, y = receiver_d_dist_preds), size=1.5) +
  annotate("segment", x=fiddy_pt, xend= fiddy_pt, y = 0, yend=fiddy_pt_pred, colour='#4292c6', alpha = 0.5, size = 1.25) + 
  annotate("segment", x=fiddy_pt, xend= fiddy_pt+2, y = fiddy_pt_pred, yend=fiddy_pt_pred,
           colour='#4292c6', alpha=0.5, size = 1.25) + 
  annotate("label", x = fiddy_pt+2, y=fiddy_pt_pred, hjust = 0,
           label=paste0("When receivers are only inches",# round(fiddy_pt,1),
                        " away from their nearest \ndefender, coverage wins about 50%, on average.")) +
  xlim(0,5) +
  labs(x = '\nRecevier Yds. From Defender', y = 'Coverage Win Rate\n',
       title = "Having defense nearby isn't the strongest effect, but it's still significant")


### Multiple effects
data_downfield$height_diff_cat = ifelse(abs(data_downfield$receiver_minus_defender_height) <= 2, "0-2 inch difference",
                                        ifelse((data_downfield$receiver_minus_defender_height > 2) & 
                                                (data_downfield$receiver_minus_defender_height < 4), "Receiver +2 to 4 inches",
                                               ifelse(data_downfield$receiver_minus_defender_height >= 4, "Receiver +4 or more inches",
                                                      ifelse((data_downfield$receiver_minus_defender_height < -2) &
                                                               (data_downfield$receiver_minus_defender_height) > -4, "Defender +2 to 4 inches",
                                                             "Defender +4 or more inches"))))

## Receiver distance to defender, controlling for height diff
receiver_d_ht_dist_fit = glm(target~ receiver_corner_dist_between + height_diff_cat, data = data_downfield, family = binomial)


data_downfield$rdhd_preds = predict(receiver_d_ht_dist_fit, type = 'response')


fiddy_pt = data_downfield$receiver_corner_dist_between[which.min(data_downfield$rdd_pred_minus_pt5)]
fiddy_pt_pred = data_downfield$receiver_d_dist_preds[which.min(data_downfield$rdd_pred_minus_pt5)]

ggplot(data_downfield, aes(x = receiver_distance_to_football, y = target)) + 
  geom_point(alpha=0.005) + 
  geom_line(aes(x=receiver_corner_dist_between, y = rdhd_preds, colour = height_diff_cat), size=1.5) +
  xlim(0,5) +
  labs(x = '\nRecevier Yds. From Defender', y = 'Coverage Win Rate\n', colour = 'Height difference',
       title = "Defenders have most success when only *slightly* taller than receivers")


### All of the effects
full_model = glm(target~ yds_thrown_past_los + receiver_distance_to_football + receiver_corner_dist_between +
                         height_diff_cat, data = data_downfield, family = binomial)


data_downfield$full_model_preds = predict(full_model, type = 'response')

### Receivers with greatest difference in predicted defense win rate %
data_downfield %>% mutate(residual = target - full_model_preds) %>% 
  group_by(receiver_name) %>% summarise(cnt= n(),
                                       avg_residual = mean(residual)) %>%
  filter(cnt > 50) %>%
  top_n(5, avg_residual) %>% arrange(desc(avg_residual))
## Lots of bigger receivers and TEs in here.. and some with drop problems.  
#  Surprising to see Kittle here, but 2018 was his worst catch %

data_downfield %>% mutate(residual = target - full_model_preds) %>% 
  group_by(receiver_name) %>% summarise(cnt= n(),
                                        avg_residual = mean(residual)) %>%
  filter(cnt > 50) %>%
  top_n(-5, avg_residual) %>% arrange(avg_residual)
## Some really good receivers in here.  Tyler Lockett probably catches a lot more than expected because of his height
### Also he had an amazing 81.4% catch percentage in 2018



### Defenders with greatest difference in predicted defense win rate %
data_downfield %>% mutate(residual = target - full_model_preds) %>% 
  group_by(defender_name) %>% summarise(cnt= n(),
                                        avg_residual = mean(residual)) %>%
  filter(cnt > 50) %>%
  top_n(10, avg_residual) %>% arrange(desc(avg_residual))
## Some really top-notch corners in here.  But what is it about them all that makes them so good?


data_downfield %>% mutate(residual = target - full_model_preds) %>% 
  group_by(defender_name) %>% summarise(cnt= n(),
                                        avg_residual = mean(residual)) %>%
  filter(cnt > 50) %>%
  top_n(-10, avg_residual) %>% arrange(avg_residual)
## Some less good corners..  Surprising to see Josh Norman who was once very highly regarded