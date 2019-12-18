from supervised import RelationLearner

learner= RelationLearner()
learner.load_word_embeddings('model.vec')
learner.load_training_set('train.tsv')
learner.load_testing_set('test.tsv')

learner.impose_project('attri','linear')
learner.impose_project('coord','linear')
learner.impose_project('event','orth_linear')
learner.impose_project('mero','orth_linear')
learner.impose_project('random','piecewise',k=4)
learner.impose_project('hyper','orth_piecewise',k=4)

learner.set_default_feature_types('subject','object','offset')

model=learner.train_model()
learner.test_model(model)