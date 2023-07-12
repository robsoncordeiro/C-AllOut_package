//---------------------------------------------------------------------------//
// Author: Guilherme Domingos Faria Silva
//---------------------------------------------------------------------------
#pragma hdrstop
#include "app.h"

#pragma argsused
int main(int argc, char* argv[]){
   TApp app;

   // Init application.
   app.Init(argv[1]);
   // Run it.
   app.Run(argv[2], argv[3], argv[4]);
   // Release resources.
   app.Done();

   return 0;
}//end main
