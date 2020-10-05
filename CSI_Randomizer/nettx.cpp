//
// Copyright 2011-2012,2014 Ettus Research LLC
// Copyright 2018 Ettus Research, a National Instruments Company
// Copyright 2020 Francesco Gringoli
//
// SPDX-License-Identifier: GPL-3.0-or-later
//

#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>
#include <fstream>
#include <complex>
#include <csignal>
#include <chrono>
#include <thread>
#include <cmath>

#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

namespace po = boost::program_options;

static bool stop_signal_called = false;
void sig_int_handler(int){stop_signal_called = true;}

template<typename samp_type> void send_from_net(
    uhd::tx_streamer::sptr tx_stream,
    uhd::usrp::multi_usrp::sptr usrp
){

    uhd::tx_metadata_t md;
    md.start_of_burst = false;
    md.end_of_burst = false; 

    int listensoc = socket(PF_INET, SOCK_STREAM, 0);
    if (listensoc == -1) {
        throw std::runtime_error("Cannot create socket.");
    }
    int reuse = 1;
    if (setsockopt(listensoc, SOL_SOCKET, SO_REUSEADDR, (const char*) &reuse, sizeof(reuse)) < 0) {
	throw std::runtime_error("Cannot set socket reuse.");
    }
    struct sockaddr_in local;
    struct sockaddr_in remote;
    local.sin_family = PF_INET;
    local.sin_port = htons((short) 12345);
    local.sin_addr.s_addr = INADDR_ANY;


    if (bind (listensoc, (struct sockaddr*) &local, sizeof(local)) == -1) {
	throw std::runtime_error ("Cannot bind socket.");
    }

    if (listen (listensoc, 1) == -1) {
	throw std::runtime_error ("Cannot activate listen.");
    }

    int longAdr = sizeof(remote);
    int connectsoc = accept(listensoc, (struct sockaddr*)&remote, (socklen_t*) &longAdr);
    int retc = 0;
    while(not stop_signal_called){
	// this should be fixed... it can read just one byte (very low chance actually)
	// in fact, we exit if this happens
	short remaining = 0;
	retc = read (connectsoc, &remaining, sizeof (remaining));
	if (retc < 2) break;
        // remaining = ntohs(remaining);
	std::cout << "receiving " << remaining << " elements" << std::endl;
        std::vector<samp_type> buff(remaining);
	int received = 0;
	int remainingbytes = remaining * sizeof(samp_type);
	while (remainingbytes > 0) {
	    retc = read (connectsoc, ((char*) &buff.front()) + received, remainingbytes);
	    if (retc == -1) break;
	    remainingbytes = remainingbytes - retc;
	    received = received + retc;
	}
	if (retc == -1) break;

        std::vector<samp_type *> banks;
        banks.push_back(&buff.front ());

        tx_stream->send(banks, remaining, md);

    }
    close (connectsoc);
    close (listensoc);
}

int UHD_SAFE_MAIN(int argc, char *argv[]){
    uhd::set_thread_priority_safe();

    //variables to be set by po
    std::string args, file, type, ant, subdev, ref, wirefmt, channel;
    size_t spb;
    double rate, freq, gain, bw, lo_off, hoptime;

    //setup the program options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("args", po::value<std::string>(&args)->default_value(""), "multi uhd device address args")
        ("type", po::value<std::string>(&type)->default_value("short"), "sample type: double, float, or short")
        ("rate", po::value<double>(&rate), "rate of outgoing samples")
	("freq", po::value<double>(&freq)->default_value(5775000000), "Frequency in Hz")
        ("lo_off", po::value<double>(&lo_off), "Offset for frontend LO in Hz (optional)")
        ("gain", po::value<double>(&gain), "gain for the RF chain")
        ("subdev", po::value<std::string>(&subdev)->default_value(""), "subdevice specification")
        ("wirefmt", po::value<std::string>(&wirefmt)->default_value("sc16"), "wire format (sc8 or sc16)")
        ("int-n", "tune USRP with integer-n tuning")
        ("bw", po::value<double>(&bw), "analog frontend filter bandwidth in Hz")
        ("ref", po::value<std::string>(&ref)->default_value("internal"), "reference source (internal, external, mimo)")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //print the help message
    if (vm.count("help")){
        std::cout << boost::format("UHD TX samples from file %s") % desc << std::endl;
        return ~0;
    }

    bool repeat = vm.count("repeat") > 0;

    // Create a usrp device
    std::cout << boost::format("Creating the usrp device...") << std::endl;
    uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(args);
    std::cout << "... done" << std::endl;

    // Select the TX subdevice first; this mapping affects all the settings.
    if (vm.count("subdev")) usrp->set_tx_subdev_spec(subdev);

    // Set the tx sample rate on all channels.
    if (not vm.count("rate")){
        std::cerr << "Please specify the sample rate with --rate" << std::endl;
        return ~0;
    }
    std::cout << boost::format("Setting TX Rate: %f Msps...") % (rate/1e6) << std::endl;
    usrp->set_tx_rate(rate);
    std::cout << boost::format("Actual TX Rate: %f Msps...") % (usrp->get_tx_rate()/1e6) << std::endl << std::endl;

    // Lock motherboard clock to internal reference source and reset time register.
    usrp->set_clock_source("internal");
    usrp->set_time_now(uhd::time_spec_t(0.0));
    std::cout << "Device timestamp set to 0.0 s." << std::endl;

    uhd::tune_request_t tune_request;
    std::cout << boost::format("Setting TX Freq1: %f MHz...") % (freq / 1e6) << std::endl;
    if(vm.count("lo_off")) tune_request = uhd::tune_request_t(freq, lo_off);
    else tune_request = uhd::tune_request_t(freq);
    if(vm.count("int-n")) tune_request.args = uhd::device_addr_t("mode_n=integer");
    usrp->set_tx_freq(tune_request, 0);

    std::cout << boost::format("Actual TX Freq0: %f MHz...") % (usrp->get_tx_freq(0)/1e6) << std::endl << std::endl;

    // set the rf gain
    if (vm.count("gain")){
        std::cout << boost::format("Setting TX Gain: %f dB...") % gain << std::endl;
        usrp->set_tx_gain(gain, 0);
        std::cout << boost::format("Actual TX Gain: %f dB...") % usrp->get_tx_gain(0) << std::endl << std::endl;
    }

    // set the analog frontend filter bandwidth
    if (vm.count("bw")){
        std::cout << boost::format("Setting TX Bandwidth: %f MHz...")
                     % (bw / 1e6)
                  << std::endl;
        usrp->set_tx_bandwidth(bw);
        std::cout << boost::format("Actual TX Bandwidth: %f MHz...")
                     % (usrp->get_tx_bandwidth() / 1e6)
                  << std::endl << std::endl;
    }

    // set the antenna
    usrp->set_tx_antenna("TX/RX", 0);

    //allow for some setup time:
    std::this_thread::sleep_for(std::chrono::seconds(1));

    //Check Ref and LO Lock detect
    std::vector<std::string> sensor_names;
    sensor_names = usrp->get_tx_sensor_names(0);
    if (std::find(sensor_names.begin(), sensor_names.end(), "lo_locked") != sensor_names.end()) {
        uhd::sensor_value_t lo_locked = usrp->get_tx_sensor("lo_locked",0);
        std::cout << boost::format("Checking TX: %s ...") % lo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }
    sensor_names = usrp->get_mboard_sensor_names(0);

    //set sigint if user wants to receive
    if(repeat){
        std::signal(SIGINT, &sig_int_handler);
        std::cout << "Press Ctrl + C to stop streaming..." << std::endl;
    }

    //create a transmit streamer
    std::string cpu_format;
    std::vector<size_t> channel_nums;
    if (type == "double") cpu_format = "fc64";
    else if (type == "float") cpu_format = "fc32";
    else if (type == "short") cpu_format = "sc16";
    else if (type == "char") cpu_format = "sc8";
    uhd::stream_args_t stream_args(cpu_format, wirefmt);
    channel_nums.push_back(0);
    stream_args.channels = channel_nums;
    uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

    //send from file
    if (type == "double") send_from_net<std::complex<double> >(tx_stream, usrp);
    else if (type == "float") send_from_net<std::complex<float> >(tx_stream, usrp);
    else if (type == "short") send_from_net<std::complex<short> >(tx_stream, usrp);
    else if (type == "char") send_from_net<std::complex<char> >(tx_stream, usrp);
    else throw std::runtime_error("Unknown type " + type);

    //finished
    std::cout << std::endl << "Done!" << std::endl << std::endl;

    return EXIT_SUCCESS;
}
