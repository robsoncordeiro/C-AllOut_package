//---------------------------------------------------------------------------
// app.cpp - Implementation of the application.
//
// To change the behavior of this application, comment and uncomment lines at
// TApp::Init() and TApp::Query().
//
// Author: Guilherme Domingos Faria Silva
//---------------------------------------------------------------------------
#include <iostream>
#pragma hdrstop
#include "app.h"

//---------------------------------------------------------------------------
#pragma package(smart_init)
//---------------------------------------------------------------------------
// Class TApp
//------------------------------------------------------------------------------
void TApp::CreateTree()
{
    // create for Slim-Tree
    SlimTree = new mySlimTree(PageManager);

    SlimTree->SetSplitMethod(stSlimTree<TCity, TCityDistanceEvaluator>::smSPANNINGTREE);
    SlimTree->SetChooseMethod(stSlimTree<TCity, TCityDistanceEvaluator>::cmMINDIST);
}//end TApp::CreateTree

//------------------------------------------------------------------------------
void TApp::CreatePageManager(char * pageSize)
{
    int treePageSize = strtol(pageSize, NULL, 0);

    // cout << "\n\nPage size: " << treePageSize << endl;

    PageManager = new stMemoryPageManager(treePageSize);
}//end TApp::CreateDiskPageManager

//------------------------------------------------------------------------------
void TApp::Run(char * fileWithOutlierScores, char * fileWithDatasetToLoad, char * cut)
{
    // Lets load the tree with a lot values from the file.
    // cout << "\n\nAdding objects in the SlimTree";
    LoadTree(fileWithDatasetToLoad, cut);

    FindOutliers(fileWithOutlierScores);
    // Hold the screen.
    //cout << "\n\nFinished the whole test!";
}//end TApp::Run

//------------------------------------------------------------------------------
void TApp::Done()
{
    if (this->SlimTree != NULL)
    {
        delete this->SlimTree;
    }//end if

    if (this->PageManager != NULL)
    {
        delete this->PageManager;
    }//end if
}//end TApp::Done

//------------------------------------------------------------------------------
void TApp::LoadTree(char * fileWithDatasetToLoad, char * cut)
{
    ifstream in(fileWithDatasetToLoad);
    string instanceName;
    string token;
    string strTmp;
    vector<double> instance;
    int instanceLength;

    long w = 0;
    TCity * object;
    int counter = 0;

    int cutNumber = strtol(cut, NULL, 0);

    if (SlimTree != NULL)
    {
        if (in.is_open())
        {
            //cout << "\nLoading objects...";

            while(getline(in, strTmp))
            {
                if (counter >= cutNumber) {
                    break;
                }

                instanceLength = 0;
                instance.clear();

                istringstream ss(strTmp);
                getline(ss, token, ';');
                instanceName = token;

                while(getline(ss, token, ';'))
                {
                    instanceLength++;
                    instance.push_back(stod(token));
                }//end while

                object = new TCity(instanceName, instance, instanceLength);
                SlimTree->Add(object);
                delete object;
                w++;
                counter++;
            }//end while
            // cout << "Added " << SlimTree->GetNumberOfObjects() << " objects ";
            in.close();
        }//end if
        else
        {
            cout << "\nProblem to open the file.";
        }//end if
    }
    else
    {
        cout << "\nNo objects added!";
    }//end if
}//end TApp::LoadTree

//------------------------------------------------------------------------------
void TApp::FindOutliers(char * fileWithOutlierScores)
{
    // cout << "\nStarting outlier search...";
    SlimTree->CalloutOutlierness(fileWithOutlierScores);
    // cout << "\nOutlier search finished.";
}//end TApp::FindOutliers
