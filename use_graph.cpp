#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

float evaluateNN(std::map<std::string, float> value_map, std::string path_to_model,
		 std::string path_to_branchname_file);


int main()
{
    
    // create test vector to feed
    std::map<std::string, float> variablemap;
    
    variablemap["Evt_CSV_Average"] = 5.733704566955566406e-01;
    variablemap["Evt_Deta_JetsAverage"] = 8.720633983612060547e-01;
    variablemap["HT"] = 4.278960876464843750e+02;
    variablemap["M3"] = 1.645960540771484375e+02;
    variablemap["MET"] = 2.717908859252929688e+01;
    variablemap["MHT"] = 1.342037105560302734e+01;
    variablemap["Mlb"] = 9.981956481933593750e+01;
    variablemap["all_sum_pt_with_met"] = 6.318931884765625000e+02;
    variablemap["aplanarity"] = 1.755445748567581177e-01;
    variablemap["avg_btag_disc_btags"] = 9.304379224777221680e-01;
    variablemap["avg_dr_tagged_jets"] = 1.876907467842102051e+00;
    variablemap["best_higgs_mass"] = 4.699427413940429688e+01;
    variablemap["closest_tagged_dijet_mass"] = 4.699427413940429688e+01;
    variablemap["dEta_fn"] = 1.902727931737899780e-01;
    variablemap["dev_from_avg_disc_btags"] = 1.416983199305832386e-03;
    variablemap["dr_between_lep_and_closest_jet"] = 8.027758598327636719e-01;
    variablemap["fifth_highest_CSV"] = 1.943878680467605591e-01;
    variablemap["first_jet_pt"] = 1.367731170654296875e+02;
    variablemap["fourth_highest_btag"] = 3.236412703990936279e-01;
    variablemap["fourth_jet_pt"] = 5.339822387695312500e+01;
    variablemap["h0"] = 3.956693410873413086e-01;
    variablemap["h1"] = -2.908740006387233734e-02;
    variablemap["h2"] = -4.715773835778236389e-02;
    variablemap["h3"] = 2.947593033313751221e-01;
    variablemap["invariant_mass_of_everything"] = 7.076216430664062500e+02;
    variablemap["lowest_btag"] = 8.888484835624694824e-01;
    variablemap["maxeta_jet_jet"] = 4.586497247219085693e-01;
    variablemap["maxeta_jet_tag"] = 2.464087605476379395e-01;
    variablemap["maxeta_tag_tag"] = 1.876368373632431030e-01;
    variablemap["min_dr_tagged_jets"] = 5.304006338119506836e-01;
    variablemap["pt_all_jets_over_E_all_jets"] = 8.485268354415893555e-01;
    variablemap["second_highest_btag"] = 9.224539399147033691e-01;
    variablemap["second_jet_pt"] = 8.660884857177734375e+01;
    variablemap["sphericity"] = 3.632729053497314453e-01;
    variablemap["tagged_dijet_mass_closest_to_125"] = 1.364051666259765625e+02;
    variablemap["third_highest_btag"] = 8.888484835624694824e-01;
    variablemap["third_jet_pt"] = 8.132450866699218750e+01;

    float output = evaluateNN(variablemap, "2x100.pb", "variable_order.txt");
    std::cout << output << std::endl;	   
  
    return 0;
}

float evaluateNN(std::map<std::string, float> value_map, std::string path_to_model,
		 std::string path_to_variablename_file) {

    // read variable names in vector
    std::string line;
    std::vector<std::string> variable_names;
    std::ifstream name_file(path_to_variablename_file);

    while (std::getline(name_file, line)) {
	variable_names.push_back(line);
    }
    name_file.close();

    // place variables in vector with the right order, vector-like variables
    // not yet supported
    std::vector<float> x;
    for (std::vector<std::string>::iterator it=variable_names.begin();
	 it!=variable_names.end(); ++it) {
	x.push_back(value_map[*it]);
    }

    // Initialize tensorflow session
    tensorflow::Session* session;
    tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
	std::cout << status.ToString() << "\n";
        return 1;
    }

    // Read in the protobuf graph
    tensorflow::GraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), path_to_model, &graph_def);
    if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 1;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
	std::cout << status.ToString() << "\n";
	return 1;
    }

    // Initialize tensor with values from x using the EIGEN representation of the tensorflow
    // tensor.
    tensorflow::Tensor event(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, x.size()}));
    auto event_tensor = event.tensor<float, 2>();
    for (int feature=0; feature<x.size(); feature++) {
	event_tensor(0, feature) = x[feature];
    }

    // string has to be the name of node in the graph we want to put the data in
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
	{ "input_node", event},
    };

    // The session will initialize the outputs
    std::vector<tensorflow::Tensor> output;

    // Run the session, evaluating our "output_node" operation from the graph
    status = session->Run(inputs, {"output_node"}, {}, &output);
    if (!status.ok()) {
	std::cout << status.ToString() << "\n";
       	return 1;
    }

    // Get the output of the neural network as float representation and return it
    auto out_EIGEN = output[0].scalar<float>();
    float out_float = out_EIGEN();

    // Free any resources used by the session
    session->Close();

    return out_float;
}
