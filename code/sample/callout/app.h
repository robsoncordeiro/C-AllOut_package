//---------------------------------------------------------------------------
// app.h - Implementation of the application.
//
// Author: Guilherme Domingos Faria Silva
//---------------------------------------------------------------------------
#ifndef appH
#define appH

// Metric Tree includes
#include <arboretum/stMetricTree.h>
#include <arboretum/stPlainDiskPageManager.h>
#include <arboretum/stDiskPageManager.h>
#include <arboretum/stMemoryPageManager.h>
#include <arboretum/stSlimTree.h>
#include <arboretum/stMetricTree.h>

// My object
#include "object.h"

#include <string.h>
#include <fstream>
//---------------------------------------------------------------------------
// class TApp
//---------------------------------------------------------------------------
class TApp{
   public:
      /**
      * This is the type used by the result.
      */
      typedef stResult < TCity > myResult;

      typedef stMetricTree < TCity, TCityDistanceEvaluator > MetricTree;

      /**
      * This is the type of the Slim-Tree defined by TCity and
      * TCityDistanceEvaluator.
      */
      typedef stSlimTree < TCity, TCityDistanceEvaluator > mySlimTree;

      /**
      * Creates a new instance of this class.
      */
      TApp(){
         PageManager = NULL;
         SlimTree = NULL;
      }//end TApp

      /**
      * Initializes the application.
      *
      * @param pageSize
      * @param minOccup
      * @param quantidade
      * @param prefix
      */
      void Init(char * pageSize){
         // Creates the page manager
         CreatePageManager(pageSize);
         // Creates the tree
         CreateTree();
      }//end Init

      /**
      * Runs the application.
      *
      * @param DataPath
      * @param DataQueryPath
      */
      void Run(char * fileWithOutlierScores, char * fileWithDatasetToLoad, char * cut);

      /**
      * Deinitialize the application.
      */
      void Done();

   private:

      /**
      * The Page Manager for SlimTree.
      */
      stMemoryPageManager * PageManager;

      /**
      * The SlimTree.
      */
      mySlimTree * SlimTree;

      /**
      * Vector for holding the query objects.
      */
      vector <TCity *> queryObjects;

      /**
      * Creates a disk page manager. It must be called before CreateTree().
      */
      void CreatePageManager(char * pageSize);

      /**
      * Creates a tree using the current PageManager.
      */
      void CreateTree();

      /**
      * Loads the tree from file with a set of objects.
      */
      void LoadTree(char * fileWithDatasetToLoad, char * cut);

      /**
      * Find the outlierness scores of every data instance.
      */
      void FindOutliers(char * fileWithOutlierScores);

      /**
      * Find the outlierness scores of a specific data instance.
      */
      void PerformOutlierPointQuery(char * fileWithOutlierScores, char * fileWithObjectToQuery);

};//end TApp

#endif //end appH
