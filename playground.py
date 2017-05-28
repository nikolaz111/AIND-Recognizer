my_testword = 'CHOCOLATE'

features = features_ground
features = features_norm
features = features_polar
features = features_delta
features = features_custom
model, logL = train_a_word(my_testword, 6, features) # Experiment here with different parameters
print(features)
show_model_stats(my_testword, model)
print("logL = {}\n".format(logL))

#features = features_norm
#model, logL = train_a_word(my_testword, 6, features)
#print(features)
#show_model_stats(my_testword, model)
#print("logL = {}\n".format(logL))
#
#features = features_polar
#model, logL = train_a_word(my_testword, 6, features)
#print(features)
#show_model_stats(my_testword, model)
#print("logL = {}\n".format(logL))
#
#features = features_delta
#model, logL = train_a_word(my_testword, 6, features)
#print(features)
#show_model_stats(my_testword, model)
#print("logL = {}\n".format(logL))
#
#features = features_custom
#model, logL = train_a_word(my_testword, 6, features)
#print(features)
#show_model_stats(my_testword, model)
#print("logL = {}\n".format(logL))