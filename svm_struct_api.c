/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <stdio.h>
#include <string.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"

#include <stdlib.h>
#define Dim 69
#define PhoneNum 48

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}


char *chomp(char *str) 
{ 
  if (*str && str[strlen(str)-1]=='\n') str[strlen(str)-1]=0; 
  return str; 
}


SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples = NULL;
  long     n;       /* number of examples */
  FILE     *train;
  ssize_t  read;
  size_t   len = 0;
  char     *line = NULL;

  char*    token;
  int      frameNum;
  /* read file */  
  train = fopen(file,"r");

  n = 0;
  examples = (EXAMPLE*)my_malloc(sizeof(EXAMPLE));

  int i = 0; // for iteration
  while( (read = getline(&line, &len, train)) != -1 ){
    line = chomp(line);
    //frameID
    //printf("%s %d\n", line, strlen(line));
    examples[n].x.frameID = (char*)my_malloc(sizeof(char)*strlen(line));
    examples[n].y.frameID = (char*)my_malloc(sizeof(char)*strlen(line));
    strcpy(examples[n].x.frameID, line);
    strcpy(examples[n].y.frameID, line);

    //frameNum
    read = getline(&line, &len, train);
    line = chomp(line);
    frameNum = atof(line);
    examples[n].x.frameNum = frameNum;
    examples[n].y.frameNum = frameNum;

    examples[n].y.phone   = (int*)my_malloc(sizeof(int)*frameNum);
    examples[n].x.feature = (float*)my_malloc(sizeof(float)*frameNum*Dim);

    //y
    read = getline(&line, &len, train);
    token = strtok(line, " ");
    i = 0;     
    //while((token != NULL) && (i < examples[n].y.frameNum)){
    for(i = 0;i < examples[n].y.frameNum;i++){
      examples[n].y.phone[i] = atoi(token);
      token = strtok(NULL, " ");
    }

    //x
    read = getline(&line, &len, train);
    token = strtok(line, " ");     
    i = 0;
    while(token != NULL){
      examples[n].x.feature[i] = atof(token);
      token = strtok(NULL, " ");
      i++;
    }    
    n++;
    examples = (EXAMPLE*)realloc(examples,sizeof(EXAMPLE)*(n+1));
  }

  fclose(train);
  sample.n=n;
  sample.examples=examples;
  return(sample);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */

  sm->sizePsi=PhoneNum*PhoneNum+PhoneNum*Dim; /* replace by appropriate number of features */
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=my_malloc(sizeof(DOC *)*sizePsi);
    c.rhs=my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }
  return(c);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label yhat for pattern x that scores the highest
     according to the linear evaluation function in sm, especially the
     weights sm.w. The returned label is taken as the prediction of sm
     for the pattern x. The weights correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. If the
     function cannot find a label, it shall return an empty label as
     recognized by the function empty_label(y). */
  LABEL y;

  /* insert your code for computing the predicted label y here */

  return(y);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar)) 

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;
printf(" find_most_violated_constraint_slackrescaling\n");
  /* insert your code for computing the label ybar here */

  return(ybar);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the margin rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)+psi(x,ybar)

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;
  /* insert your code for computing the label ybar here */
  int i,j,k,f,index;
  float Value[PhoneNum];
  float obserValue[PhoneNum];
  float temp,transValue,maxValue;
  VPATH **node;
  int   pre;

  node = (VPATH**)my_malloc(sizeof(VPATH*)*y.frameNum);
  for(i = 0; i < y.frameNum; i++){
    node[i] = (VPATH*)my_malloc(sizeof(VPATH)*PhoneNum);
  }

  //start Viterbi
  for(i = 0; i < PhoneNum; i++){
    node[0][i].pre = i;
    node[0][i].label = i;
    node[0][i].score = 0;
  }

  //find viterbi path
  for(f = 1; f < y.frameNum; f++){
    //observation
    for(i = 0; i < PhoneNum; i++){
      for(j = 0; j < Dim; j++){
        obserValue[i]+= x.feature[j+(f-1)*Dim]*(sm->w[j+i*Dim+1]);
      }
    }
    //transition
    for(i = 0; i < PhoneNum; i++){
      node[f][i].label = i;
      for(j = 0; j < PhoneNum; j++){
        transValue = node[f-1][j].score + sm->w[PhoneNum*(Dim+i)+j+1];
        Value[j] = transValue + obserValue[j];
      }
      temp = Value[0];
      for(j = 0; j < PhoneNum; j++){
        if(temp <= Value[j]){
          temp = Value[j];
          index = j;
        }
      }
      maxValue = temp;

      node[f][i].pre = index;
      node[f][i].score = maxValue;
      //loss
      if(i != y.phone[f]){
        node[f][i].score += 1.0;      
      }
    }
  }

  //end viterbi
  //printf("end viterbi\n");
  maxValue = node[y.frameNum-1][0].score;
  for(i = 0; i < PhoneNum; i++){
    if(maxValue < node[y.frameNum-1][i].score){
      maxValue = node[y.frameNum-1][i].score;
      index = i;
    }
  }

  //trance back
  ybar.frameID = (char*)my_malloc(sizeof(char)*strlen(y.frameID));
  strcpy(ybar.frameID, y.frameID);
  ybar.phone = (int*)my_malloc(sizeof(int)*y.frameNum);
  ybar.frameNum = y.frameNum;
  pre = node[y.frameNum-1][index].pre;
  //ybar.phone[y.frameNum-1] = index;
  for(f = y.frameNum-1; f >=0; f--){
    index = node[f][pre].label;
    ybar.phone[f] = index;
    if(f>0)pre = node[f][index].pre; 
  }
  for(i = 0; i < y.frameNum; i++)
    free(node[i]);
  free(node);

  return(ybar);
}

int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */
  return(0);
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
  SVECTOR *fvec=NULL;
  WORD* words;
  int i,j,trans;
  int SIZE = sm->sizePsi;
  float **tempfeature = (float**)my_malloc(sizeof(float*)*PhoneNum);
  float *transfeature = (float*)my_malloc(sizeof(float)*PhoneNum*PhoneNum);
  for(i = 0; i < PhoneNum; i++){
    tempfeature[i] = my_malloc(sizeof(float)*Dim);
    for(j = 0; j < Dim; j++){
      tempfeature[i][j] = 0.00;
    }
  } 
  for(i = 0; i < PhoneNum*PhoneNum; i++)   
    transfeature[i] = 0;

  /* insert code for computing the feature vector for x and y here */

  for(i = 0; i < y.frameNum-1; i++){
    trans = y.phone[i]*PhoneNum + y.phone[i+1];
    transfeature[trans]+=1.0;
  }

  for(i = 0; i < y.frameNum; i++){
    for(j = 0; j < Dim; j++){
      tempfeature[ y.phone[i] ][j] += x.feature[j+i*Dim];
    }
  }

  words = (WORD*)my_malloc(sizeof(WORD)*SIZE+1);
  
  for(i = 0; i < SIZE; i++){
    words[i].wnum = i + 1;
    if( i < PhoneNum*Dim ){
      j = floor(i/Dim); 
      words[i].weight = tempfeature[j][i - j*Dim];
    }
    else{
      words[i].weight = transfeature[i-PhoneNum*Dim];
    }
  }
  words[SIZE].wnum = 0;
  //words[SIZE].weight = 0; //terminator
  fvec=create_svector(words,"",1.0);

  //free
  free(transfeature);
  for(i = 0; i < PhoneNum; i++)
    free(tempfeature[i]);
  free(tempfeature);
  free(words);
//for(i = 0; i < SIZE; i++) printf("fvec_%d: %f \n",i,fvec->words[i].weight);

  return(fvec);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
  double error = 0;
  int i;
  if(sparm->loss_function == 0) { /* type 0 loss: 0/1 loss */
    for(i = 0; i < y.frameNum; i++){
      if(y.phone[i] != ybar.phone[i])
        error = error + 1.0;
    }
  }
  else {
    /* Put your code for different loss functions here. But then
       find_most_violated_constraint_???(x, y, sm) has to return the
       highest scoring label with the largest loss. */
  }
  //printf("call LOSS, error =%f \n",error);
  return error;
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
  }
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
  FILE *modelfile;
  
  modelfile = fopen(file,"w");
  
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
} 

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

