/*
 * Deep Learning Based Detection of Interest Flooding Attacks in NDN
 * C++ Snippets for ns-3 / ndnSIM Simulation
 * 
 * Note: These are pseudocode/snippets intended to show how to configure 
 * normal traffic and various attack vectors in an ndnSIM application.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/ndnSIM-module.h"

using namespace ns3;

int main(int argc, char* argv[]) {
    CommandLine cmd;
    cmd.Parse(argc, argv);

    // 1. Create nodes and topology
    AnnotatedTopologyReader topologyReader("", 25);
    topologyReader.SetFileName("src/ndnSIM/examples/topologies/topo-grid-3x3.txt");
    topologyReader.Read();

    // 2. Install NDN stack on all nodes
    ndn::StackHelper ndnHelper;
    ndnHelper.SetDefaultRoutes(true);
    ndnHelper.InstallAll();

    /* ==============================================================
     * A. NORMAL NDN TRAFFIC SIMULATION
     * ==============================================================
     * Concept: Legitimate consumers request popular and varying content.
     * Expected Network Effect: Balanced Interest and Data rates, PIT sizes are managed.
     */
    ndn::AppHelper normalConsumerHelper("ns3::ndn::ConsumerCbr");
    normalConsumerHelper.SetPrefix("/prefix/normal_data");
    normalConsumerHelper.SetAttribute("Frequency", StringValue("100")); // 100 interests per second
    // Install consumer on some node
    ApplicationContainer normalApp = normalConsumerHelper.Install(Names::Find<Node>("Node0"));
    normalApp.Start(Seconds(1.0));
    normalApp.Stop(Seconds(50.0));

    // Normal Producer
    ndn::AppHelper producerHelper("ns3::ndn::Producer");
    producerHelper.SetPrefix("/prefix");
    producerHelper.SetAttribute("PayloadSize", StringValue("1024"));
    ApplicationContainer producerApp = producerHelper.Install(Names::Find<Node>("Node8"));
    producerApp.Start(Seconds(0.0));

    /* ==============================================================
     * B. INTEREST FLOODING ATTACK (IFA)
     * ==============================================================
     * Concept: Attacker floods network with interests for non-existent content at high frequency.
     * Expected Network Effect: Massive PIT accumulation, dropping legitimate requests, satisfaction ratio drops near 0.
     */
    ndn::AppHelper ifaAttackerHelper("ns3::ndn::ConsumerCbr");
    // Prefix is configured such that no producer serves it, generating dynamic names
    ifaAttackerHelper.SetPrefix("/prefix/non_existent_data");
    ifaAttackerHelper.SetAttribute("Frequency", StringValue("500")); // Very high frequency
    ApplicationContainer ifaApp = ifaAttackerHelper.Install(Names::Find<Node>("Node2"));
    ifaApp.Start(Seconds(10.0));
    ifaApp.Stop(Seconds(30.0));


    /* ==============================================================
     * C. NEW ATTACK 1: SLOW INTEREST FLOODING ATTACK
     * ==============================================================
     * Concept: Attacker sends malicious interests at a stealthy, low rate just below the detection threshold.
     * Expected Network Effect: Gradual degradation of network performance, high false negative rate for threshold-based systems, PIT slowly fills with long timeouts.
     */
    ndn::AppHelper slowIfaAttacker("ns3::ndn::ConsumerCbr");
    slowIfaAttacker.SetPrefix("/prefix/stealthy_data_request");
    slowIfaAttacker.SetAttribute("Frequency", StringValue("30")); // Low, stealthy frequency
    // Modify Interest packets to have maximum lifetime
    slowIfaAttacker.SetAttribute("LifeTime", StringValue("10s")); // Keep PIT entry alive for excessively long
    ApplicationContainer slowIfaApp = slowIfaAttacker.Install(Names::Find<Node>("Node3"));
    slowIfaApp.Start(Seconds(15.0));
    slowIfaApp.Stop(Seconds(45.0));


    /* ==============================================================
     * D. NEW ATTACK 2: CACHE POLLUTION ATTACK (Locality Disruption)
     * ==============================================================
     * Concept: Attacker continuously requests unpopular valid content to force routers to cache them.
     * Expected Network Effect: Content Store (CS) fills with garbage data, legitimate users face Cache Misses, increasing network delay and latency.
     */
    ndn::AppHelper cachePollutionAttacker("ns3::ndn::ConsumerZipfMandelbrot");
    cachePollutionAttacker.SetPrefix("/prefix/unpopular_data");
    // Zipf parameter altered to target the "tail" of the distribution (unpopular content)
    cachePollutionAttacker.SetAttribute("Frequency", StringValue("100"));
    cachePollutionAttacker.SetAttribute("q", StringValue("0.0")); 
    cachePollutionAttacker.SetAttribute("s", StringValue("0.1")); // Low skewness to request diverse unpopular content
    ApplicationContainer pollutionApp = cachePollutionAttacker.Install(Names::Find<Node>("Node4"));
    pollutionApp.Start(Seconds(5.0));
    pollutionApp.Stop(Seconds(40.0));


    // 3. Configure logging Strategy
    ndn::L3RateTracer::InstallAll("dataset/traffic_logs.txt", Seconds(1.0));
    ndn::CsTracer::InstallAll("dataset/cs_logs.txt", Seconds(1.0));

    /* ==============================================================
     * E. NEW ATTACK 3: DISTRIBUTED INTEREST FLOODING (D-IFA)
     * ==============================================================
     * Concept: Multiple attacker nodes generate interests at a moderate rate, combining to overwhelm PITs.
     */
    ndn::AppHelper distAttacker("ns3::ndn::ConsumerCbr");
    distAttacker.SetPrefix("/prefix/distributed_attack");
    distAttacker.SetAttribute("Frequency", StringValue("150")); 
    ApplicationContainer distApp1 = distAttacker.Install(Names::Find<Node>("Node5"));
    ApplicationContainer distApp2 = distAttacker.Install(Names::Find<Node>("Node6"));
    distApp1.Start(Seconds(20.0)); distApp1.Stop(Seconds(40.0));
    distApp2.Start(Seconds(20.0)); distApp2.Stop(Seconds(40.0));

    /* ==============================================================
     * F. NEW ATTACK 4: PULSING INTEREST FLOODING
     * ==============================================================
     * Concept: Bursts of extreme high frequency requests followed by pauses to evade simple rate limiters.
     */
    ndn::AppHelper pulsingAttacker("ns3::ndn::ConsumerCbr");
    pulsingAttacker.SetPrefix("/prefix/pulsing_attack");
    pulsingAttacker.SetAttribute("Frequency", StringValue("800"));
    ApplicationContainer pulsingApp = pulsingAttacker.Install(Names::Find<Node>("Node7"));
    // Pulse 1
    pulsingApp.Start(Seconds(5.0)); pulsingApp.Stop(Seconds(8.0));
    // Pulse 2
    pulsingApp.Start(Seconds(20.0)); pulsingApp.Stop(Seconds(23.0));
    // Pulse 3
    pulsingApp.Start(Seconds(40.0)); pulsingApp.Stop(Seconds(43.0));

    // Run Simulation
    Simulator::Stop(Seconds(60.0));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
