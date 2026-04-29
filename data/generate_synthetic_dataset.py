#!/usr/bin/env python3
"""
generate_synthetic_dataset.py — Generate a realistic synthetic telecom log dataset
for evaluating RAG systems on multi-hop root cause analysis.

Generates:
    - Multiple log files (eNodeB, CU-CP, CU-UP, RAN, Core Network, Transport)
    - 800-1000 log entries per scenario
    - Realistic fields: timestamp, module, error_code, severity, message
    - Multi-step failure chains across files
    - 50+ evaluation queries with ground truth

Output:
    - data/synthetic_logs/*.txt (log files)
    - data/synthetic_eval_queries.json (queries + ground truth)
"""

import json
import os
import random
import datetime
from pathlib import Path
from typing import List, Dict, Tuple

random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "synthetic_logs"
QUERIES_OUTPUT = Path(__file__).parent / "synthetic_eval_queries.json"

# ─── Telecom Modules and Components ──────────────────────────────────────────

MODULES = {
    "enodeb": ["RRC", "PDCP", "RLC", "MAC", "PHY", "S1AP", "X2AP", "OAM"],
    "cu_cp": ["F1AP", "NGAP", "RRC", "UE_CONTEXT", "BEARER_MGMT", "HANDOVER", "PAGING"],
    "cu_up": ["GTP-U", "SDAP", "PDCP", "BEARER", "QOS", "PACKET_PROC"],
    "ran": ["CELL_MGMT", "RADIO_RESOURCE", "SCHEDULER", "POWER_CTRL", "INTERFERENCE"],
    "core_nw": ["AMF", "SMF", "UPF", "NRF", "AUSF", "UDM", "PCF"],
    "transport": ["SCTP", "GTP", "IPSec", "VLAN", "BACKHAUL", "FRONTHAUL"],
}

ERROR_CODES = {
    "RRC": ["RRC_CONN_REJ_001", "RRC_RECONFIG_FAIL_004", "RRC_RELEASE_ABNORMAL_007", 
             "RRC_SETUP_TIMEOUT_003", "RRC_INTEGRITY_FAIL_009"],
    "F1AP": ["F1AP_UE_CTX_SETUP_FAIL_012", "F1AP_RESET_015", "F1AP_MSG_DECODE_ERR_018"],
    "NGAP": ["NGAP_INIT_CTX_FAIL_021", "NGAP_PATH_SWITCH_FAIL_024", "NGAP_HANDOVER_FAIL_027"],
    "GTP-U": ["GTPU_TUNNEL_ERR_031", "GTPU_SEQ_MISMATCH_034", "GTPU_PATH_FAIL_037"],
    "PDCP": ["PDCP_INTEGRITY_FAIL_041", "PDCP_CIPHER_FAIL_044", "PDCP_SN_OVERFLOW_047"],
    "RLC": ["RLC_MAX_RETX_051", "RLC_REASSEMBLY_TIMEOUT_054", "RLC_STATUS_ERR_057"],
    "MAC": ["MAC_RACH_FAIL_061", "MAC_BSR_TIMEOUT_064", "MAC_HARQ_NACK_MAX_067"],
    "S1AP": ["S1AP_PATH_FAIL_071", "S1AP_RESET_074", "S1AP_OVERLOAD_077"],
    "SCTP": ["SCTP_ASSOC_FAIL_081", "SCTP_HEARTBEAT_TIMEOUT_084", "SCTP_CHUNK_ERR_087"],
    "AMF": ["AMF_REG_REJECT_091", "AMF_AUTH_FAIL_094", "AMF_UE_CTX_REL_097"],
    "SMF": ["SMF_PDU_SESS_FAIL_101", "SMF_QOS_FLOW_ERR_104", "SMF_BEARER_MOD_FAIL_107"],
    "UPF": ["UPF_TUNNEL_ERR_111", "UPF_BUFF_OVERFLOW_114", "UPF_RULE_CONFLICT_117"],
    "BEARER_MGMT": ["BEARER_SETUP_FAIL_121", "BEARER_MOD_TIMEOUT_124", "BEARER_REL_ERR_127"],
    "SCHEDULER": ["SCHED_DL_FAIL_131", "SCHED_UL_TIMEOUT_134", "SCHED_RESOURCE_EXHAUST_137"],
    "HANDOVER": ["HO_PREP_FAIL_141", "HO_EXEC_FAIL_144", "HO_CANCEL_147"],
}

SEVERITY_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# ─── Failure Scenario Templates ──────────────────────────────────────────────

class FailureScenario:
    """Defines a multi-step failure chain across log files."""
    
    def __init__(self, scenario_id: str, name: str, description: str,
                 root_cause: str, affected_ues: List[str],
                 chain: List[Dict], duration_ms: int = 5000):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.root_cause = root_cause
        self.affected_ues = affected_ues
        self.chain = chain  # list of {file, module, severity, error_code, message, delay_ms}
        self.duration_ms = duration_ms


def build_scenarios() -> List[FailureScenario]:
    """Create diverse failure scenarios with multi-hop causal chains."""
    
    scenarios = [
        # ─── Scenario 1: RRC Reconfiguration failure chain ────────────────
        FailureScenario(
            scenario_id="S01",
            name="RRC Reconfiguration Failure Cascade",
            description="UE7 RRC reconfiguration fails due to invalid measurement config, triggering CU-CP release and packet loss",
            root_cause="Invalid measurement configuration in RRC Reconfiguration message causes failure code 4 in rrc_reconfig_handler.cpp line 142, propagating to CU-CP UE context release and downstream GTP-U tunnel teardown with 23 packets lost",
            affected_ues=["UE7"],
            chain=[
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "INFO", "error_code": "", "message": "UE7 RRC Connected state entered, cellId=12, rnti=0x4A7B", "delay_ms": 0},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "INFO", "error_code": "", "message": "UE7 Measurement Configuration being applied, measId=3, reportConfigId=2", "delay_ms": 50},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "WARNING", "error_code": "", "message": "UE7 RRCReconfiguration message sent, size=47 bytes, containing measConfig update", "delay_ms": 100},
                {"file": "cu_cp_f1ap.log", "module": "F1AP", "severity": "INFO", "error_code": "", "message": "[ueId:7] DL RRC Message Transfer initiated, DCCH logical channel, msgSize=47", "delay_ms": 120},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "ERROR", "error_code": "RRC_RECONFIG_FAIL_004", "message": "UE7 rrc_reconfig_handler.cpp[142]: Failure (code 4) applying RRCReconfiguration - invalid measurement bandwidth for serving cell frequency", "delay_ms": 200},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "ERROR", "error_code": "RRC_RECONFIG_FAIL_004", "message": "UE7 RrcFsm::OnRrcReconfigurationFailure() invoked, transitioning to RRC_IDLE", "delay_ms": 210},
                {"file": "cu_cp_f1ap.log", "module": "UE_CONTEXT", "severity": "ERROR", "error_code": "F1AP_UE_CTX_SETUP_FAIL_012", "message": "[ueId:7] RRC Reconfiguration failure notification received from DU, cause=radioNetwork-unspecified", "delay_ms": 250},
                {"file": "cu_cp_f1ap.log", "module": "UE_CONTEXT", "severity": "CRITICAL", "error_code": "", "message": "[ueId:7] Triggering UE Context Release, cause=rrc-reconfiguration-failure, releasing all bearers", "delay_ms": 300},
                {"file": "cu_cp_f1ap.log", "module": "BEARER_MGMT", "severity": "ERROR", "error_code": "BEARER_REL_ERR_127", "message": "[ueId:7] DRB Release for bearerId=1,2,3 - data radio bearer teardown initiated", "delay_ms": 350},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "ERROR", "error_code": "GTPU_TUNNEL_ERR_031", "message": "[ueId:7] GTP-U tunnel teardown, teid=0x0000F7A1, buffered packets=23, packets_dropped=23", "delay_ms": 400},
                {"file": "cu_up_gtpu.log", "module": "PACKET_PROC", "severity": "CRITICAL", "error_code": "", "message": "[ueId:7] PacketProcessor: forward jump detected, seqNum gap 156->179 (delta=23), data loss confirmed", "delay_ms": 420},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "WARNING", "error_code": "AMF_UE_CTX_REL_097", "message": "[ueId:7] UE Context Release Command sent to gNB, cause=nas-normal-release", "delay_ms": 500},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:7] UE Context Release Complete received, UE deregistered from serving cell", "delay_ms": 600},
            ],
            duration_ms=600,
        ),
        
        # ─── Scenario 2: SCTP Association Failure ─────────────────────────
        FailureScenario(
            scenario_id="S02",
            name="SCTP Transport Link Failure",
            description="SCTP heartbeat timeout causes F1AP/NGAP path failure affecting multiple UEs",
            root_cause="SCTP heartbeat timeout on transport link between CU-CP and DU (IP 10.0.1.15:38412) due to network congestion causes F1AP association failure, leading to all UEs on that link losing connectivity with 45 UEs affected and forced re-establishment",
            affected_ues=["UE1", "UE2", "UE3", "UE4", "UE5", "UE12", "UE15", "UE18", "UE21"],
            chain=[
                {"file": "transport_sctp.log", "module": "SCTP", "severity": "WARNING", "error_code": "", "message": "SCTP association id=3 (10.0.1.15:38412): heartbeat response delayed, rto=1200ms exceeds threshold", "delay_ms": 0},
                {"file": "transport_sctp.log", "module": "SCTP", "severity": "WARNING", "error_code": "", "message": "SCTP association id=3: retransmission count=4, approaching max_retrans=5", "delay_ms": 1000},
                {"file": "transport_sctp.log", "module": "SCTP", "severity": "ERROR", "error_code": "SCTP_HEARTBEAT_TIMEOUT_084", "message": "SCTP association id=3 (10.0.1.15:38412): HEARTBEAT_TIMEOUT, max retransmissions exceeded, path failure declared", "delay_ms": 2000},
                {"file": "transport_sctp.log", "module": "SCTP", "severity": "CRITICAL", "error_code": "SCTP_ASSOC_FAIL_081", "message": "SCTP association id=3: COMMUNICATION_LOST event, peer endpoint unreachable, 45 streams affected", "delay_ms": 2100},
                {"file": "cu_cp_f1ap.log", "module": "F1AP", "severity": "CRITICAL", "error_code": "F1AP_RESET_015", "message": "F1AP: Transport layer failure detected on association to DU (10.0.1.15), initiating F1 Reset", "delay_ms": 2200},
                {"file": "cu_cp_f1ap.log", "module": "UE_CONTEXT", "severity": "ERROR", "error_code": "", "message": "Mass UE Context Release: 45 UEs affected by F1 path failure, releasing all contexts on DU-id=2", "delay_ms": 2300},
                {"file": "cu_cp_f1ap.log", "module": "UE_CONTEXT", "severity": "ERROR", "error_code": "", "message": "[ueId:1,2,3,4,5,12,15,18,21...] Batch UE Context Release initiated, cause=transport-resource-unavailable", "delay_ms": 2350},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "ERROR", "error_code": "", "message": "NGAP: Received multiple UE Context Release Requests from gNB, count=45, cause=transport-failure", "delay_ms": 2500},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "WARNING", "error_code": "", "message": "AMF: Bulk UE deregistration processing, triggering paging for voice-active UEs (count=12)", "delay_ms": 2700},
                {"file": "ran_cell.log", "module": "CELL_MGMT", "severity": "CRITICAL", "error_code": "", "message": "Cell-12: Operational state degraded, 45 UEs disconnected, initiating cell recovery procedure", "delay_ms": 2800},
                {"file": "ran_cell.log", "module": "RADIO_RESOURCE", "severity": "WARNING", "error_code": "", "message": "Cell-12: RACH load spike detected, 38 UEs attempting re-establishment simultaneously", "delay_ms": 3500},
            ],
            duration_ms=3500,
        ),
        
        # ─── Scenario 3: Handover Failure with Data Loss ─────────────────
        FailureScenario(
            scenario_id="S03",
            name="Inter-gNB Handover Failure",
            description="Handover preparation fails due to target cell overload, causing RLF and data interruption",
            root_cause="Handover preparation failure for UE11 from Cell-12 to Cell-15 due to target cell resource exhaustion (PRB utilization 97%), triggering Radio Link Failure with T310 timer expiry at the source, PDCP data loss of 847 SDUs during 2.3s interruption",
            affected_ues=["UE11"],
            chain=[
                {"file": "ran_cell.log", "module": "RADIO_RESOURCE", "severity": "INFO", "error_code": "", "message": "UE11: A3 measurement event triggered, serving RSRP=-98dBm, neighbor Cell-15 RSRP=-87dBm, offset=3dB", "delay_ms": 0},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "INFO", "error_code": "", "message": "UE11: Measurement Report received, triggering handover evaluation to Cell-15 (PCI=342)", "delay_ms": 50},
                {"file": "cu_cp_f1ap.log", "module": "HANDOVER", "severity": "INFO", "error_code": "", "message": "[ueId:11] Handover preparation initiated, source=Cell-12 target=Cell-15, cause=a3-event", "delay_ms": 100},
                {"file": "cu_cp_f1ap.log", "module": "HANDOVER", "severity": "WARNING", "error_code": "", "message": "[ueId:11] Handover Request sent to target gNB, awaiting Handover Request Acknowledge", "delay_ms": 150},
                {"file": "ran_cell.log", "module": "SCHEDULER", "severity": "WARNING", "error_code": "SCHED_RESOURCE_EXHAUST_137", "message": "Cell-15: PRB utilization at 97%, DL scheduling congestion, admission control rejecting new bearers", "delay_ms": 200},
                {"file": "cu_cp_f1ap.log", "module": "HANDOVER", "severity": "ERROR", "error_code": "HO_PREP_FAIL_141", "message": "[ueId:11] Handover Preparation Failure received from target, cause=no-radio-resources-available-in-target-cell", "delay_ms": 350},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "WARNING", "error_code": "", "message": "UE11: Handover cancelled, reverting to source cell configuration, RSRP continuing to degrade", "delay_ms": 400},
                {"file": "enodeb_rrc.log", "module": "PHY", "severity": "ERROR", "error_code": "", "message": "UE11: Consecutive HARQ NACKs on PDSCH, CQI=3 (poor), RSRP=-108dBm, approaching T310 threshold", "delay_ms": 800},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "CRITICAL", "error_code": "", "message": "UE11: T310 timer expired (1000ms), Radio Link Failure declared, initiating RRC re-establishment", "delay_ms": 1800},
                {"file": "enodeb_rrc.log", "module": "PDCP", "severity": "ERROR", "error_code": "PDCP_SN_OVERFLOW_047", "message": "UE11: PDCP data loss during RLF, 847 SDUs discarded from transmission buffer, interruption duration=2.3s", "delay_ms": 1900},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "WARNING", "error_code": "", "message": "[ueId:11] GTP-U: DL data buffering during RRC re-establishment, buffer occupancy=78%", "delay_ms": 2000},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "INFO", "error_code": "", "message": "UE11: RRC Re-establishment successful on Cell-12, resuming data transfer", "delay_ms": 2300},
            ],
            duration_ms=2300,
        ),
        
        # ─── Scenario 4: PDU Session Establishment Failure ────────────────
        FailureScenario(
            scenario_id="S04",
            name="PDU Session Establishment Rejection",
            description="SMF rejects PDU session due to QoS flow conflict, UE unable to establish data bearer",
            root_cause="SMF rejects PDU Session Establishment for UE22 because requested QoS flow (5QI=1, voice) conflicts with existing session policy (max 2 GBR flows per UE) set by PCF, resulting in NAS rejection cause #29 (user authentication/authorization failure) sent to UE",
            affected_ues=["UE22"],
            chain=[
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:22] NAS: PDU Session Establishment Request received, sessionId=3, sst=1, sd=0x010203", "delay_ms": 0},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:22] Nsmf_PDUSession_CreateSMContext Request sent to SMF, dnn=ims, s-nssai={sst:1,sd:010203}", "delay_ms": 50},
                {"file": "core_nw_smf.log", "module": "SMF", "severity": "INFO", "error_code": "", "message": "[ueId:22] PDU Session context creation initiated, pduSessionId=3, pduType=IPv4v6", "delay_ms": 100},
                {"file": "core_nw_smf.log", "module": "SMF", "severity": "INFO", "error_code": "", "message": "[ueId:22] Npcf_SMPolicyControl_Create: requesting QoS authorization from PCF", "delay_ms": 150},
                {"file": "core_nw_pcf.log", "module": "PCF", "severity": "WARNING", "error_code": "", "message": "[ueId:22] Policy decision: UE already has 2 GBR flows active (sessId=1:5QI=1, sessId=2:5QI=2), max GBR limit reached", "delay_ms": 200},
                {"file": "core_nw_pcf.log", "module": "PCF", "severity": "ERROR", "error_code": "", "message": "[ueId:22] SM Policy Create response: REJECT, cause=quota-reached, maxGbrFlows=2, currentGbrFlows=2", "delay_ms": 220},
                {"file": "core_nw_smf.log", "module": "SMF", "severity": "ERROR", "error_code": "SMF_QOS_FLOW_ERR_104", "message": "[ueId:22] PDU Session Establishment rejected: QoS flow authorization denied by PCF, 5QI=1 (GBR) exceeds policy limit", "delay_ms": 250},
                {"file": "core_nw_smf.log", "module": "SMF", "severity": "ERROR", "error_code": "SMF_PDU_SESS_FAIL_101", "message": "[ueId:22] Nsmf_PDUSession_CreateSMContext Response: FAILURE, cause=#29 user-authentication-authorization-failure", "delay_ms": 280},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "WARNING", "error_code": "", "message": "[ueId:22] NAS: PDU Session Establishment Reject sent to UE, cause=#29, backoff-timer=60s", "delay_ms": 350},
                {"file": "cu_cp_f1ap.log", "module": "NGAP", "severity": "INFO", "error_code": "", "message": "[ueId:22] Downlink NAS Transport: PDU Session Establishment Reject, msgSize=24 bytes", "delay_ms": 380},
            ],
            duration_ms=380,
        ),
        
        # ─── Scenario 5: RLC Max Retransmission / DRB Failure ─────────────
        FailureScenario(
            scenario_id="S05",
            name="RLC Max Retransmission causing DRB Failure",
            description="Persistent radio conditions cause RLC max retransmission for UE5, triggering DRB failure and re-establishment",
            root_cause="Poor radio channel conditions (SINR=-2dB, BLER=32%) for UE5 on DRB-2 cause RLC AM max retransmissions (32 attempts) at rlc_am_entity.cpp line 287, triggering RLC failure indication to RRC, DRB failure report to CU-CP, and bearer modification with reduced QoS",
            affected_ues=["UE5"],
            chain=[
                {"file": "ran_cell.log", "module": "INTERFERENCE", "severity": "WARNING", "error_code": "", "message": "Cell-12 sector-2: Uplink interference detected, IoT=+8dB above threshold on PRB group 15-25", "delay_ms": 0},
                {"file": "enodeb_rrc.log", "module": "PHY", "severity": "WARNING", "error_code": "", "message": "UE5: DL SINR degraded to -2dB, CQI=2, rank=1, BLER=32% on scheduled PRBs", "delay_ms": 200},
                {"file": "enodeb_rrc.log", "module": "MAC", "severity": "WARNING", "error_code": "MAC_HARQ_NACK_MAX_067", "message": "UE5: HARQ process 3, consecutive NACKs=8 on DRB-2, triggering RLC retransmission", "delay_ms": 500},
                {"file": "enodeb_rrc.log", "module": "RLC", "severity": "WARNING", "error_code": "", "message": "UE5: RLC AM entity DRB-2, retransmission count=16/32, polling triggered for status report", "delay_ms": 1000},
                {"file": "enodeb_rrc.log", "module": "RLC", "severity": "ERROR", "error_code": "RLC_MAX_RETX_051", "message": "UE5: rlc_am_entity.cpp[287]: RLC AM max retransmission reached (32) on DRB-2, SN=0x1A4F, bearer failure declared", "delay_ms": 2000},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "ERROR", "error_code": "", "message": "UE5: RLC Failure Indication received for DRB-2, initiating DRB failure report to CU-CP", "delay_ms": 2050},
                {"file": "cu_cp_f1ap.log", "module": "BEARER_MGMT", "severity": "ERROR", "error_code": "BEARER_SETUP_FAIL_121", "message": "[ueId:5] DRB Failure Indication: DRB-2 failed at DU, cause=rlc-max-retransmissions", "delay_ms": 2100},
                {"file": "cu_cp_f1ap.log", "module": "BEARER_MGMT", "severity": "WARNING", "error_code": "", "message": "[ueId:5] Bearer Modification initiated: DRB-2 QoS downgrade from 5QI=7 to 5QI=9 (non-GBR)", "delay_ms": 2200},
                {"file": "cu_up_gtpu.log", "module": "QOS", "severity": "WARNING", "error_code": "", "message": "[ueId:5] QoS flow remapping: flowId=2 moved from DRB-2 (failed) to DRB-1, degraded service", "delay_ms": 2300},
                {"file": "enodeb_rrc.log", "module": "RRC", "severity": "INFO", "error_code": "", "message": "UE5: RRC Reconfiguration for DRB modification sent, removing DRB-2, remapping to DRB-1", "delay_ms": 2400},
            ],
            duration_ms=2400,
        ),
        
        # ─── Scenario 6: GTP-U Path Failure (Backhaul) ───────────────────
        FailureScenario(
            scenario_id="S06",
            name="GTP-U Path Failure on Backhaul",
            description="Backhaul link degradation causes GTP-U path failure between CU-UP and UPF affecting multiple UE data sessions",
            root_cause="Backhaul fiber link degradation (BER=1e-6, latency spike to 45ms) between CU-UP and UPF (10.0.2.50) causes GTP-U Echo Request timeout after 3 retries, declaring path failure for 28 active tunnels, triggering indirect forwarding and 3.2s service interruption for affected UEs",
            affected_ues=["UE3", "UE8", "UE14", "UE19", "UE25"],
            chain=[
                {"file": "transport_sctp.log", "module": "BACKHAUL", "severity": "WARNING", "error_code": "", "message": "Backhaul link eth2 (CU-UP to UPF): latency spike detected, RTT=45ms (normal: 2ms), BER=1e-6", "delay_ms": 0},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "WARNING", "error_code": "", "message": "GTP-U Echo Request to UPF (10.0.2.50): no response, retry 1/3, timeout=3000ms", "delay_ms": 3000},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "WARNING", "error_code": "", "message": "GTP-U Echo Request to UPF (10.0.2.50): no response, retry 2/3, timeout=3000ms", "delay_ms": 6000},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "ERROR", "error_code": "GTPU_PATH_FAIL_037", "message": "GTP-U Echo Request to UPF (10.0.2.50): TIMEOUT, retry 3/3 exhausted, declaring path failure", "delay_ms": 9000},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "CRITICAL", "error_code": "GTPU_TUNNEL_ERR_031", "message": "GTP-U Path Failure: 28 active tunnels affected on path to UPF (10.0.2.50), activating indirect forwarding", "delay_ms": 9100},
                {"file": "cu_up_gtpu.log", "module": "PACKET_PROC", "severity": "ERROR", "error_code": "", "message": "[ueId:3,8,14,19,25] Data plane interrupted, buffering DL packets, buffer utilization=89%", "delay_ms": 9200},
                {"file": "cu_cp_f1ap.log", "module": "BEARER_MGMT", "severity": "WARNING", "error_code": "", "message": "Path failure notification from CU-UP: 28 bearers affected, initiating path switch for UEs [3,8,14,19,25,...]", "delay_ms": 9300},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "WARNING", "error_code": "", "message": "NGAP: Path Switch Request received from gNB for 28 UEs, processing N2 path update", "delay_ms": 9500},
                {"file": "transport_sctp.log", "module": "BACKHAUL", "severity": "INFO", "error_code": "", "message": "Backhaul link eth2: failover to redundant path eth3 completed, RTT=3ms, service resuming", "delay_ms": 12200},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "INFO", "error_code": "", "message": "GTP-U path restored via alternate route, flushing 1247 buffered packets, interruption duration=3.2s", "delay_ms": 12300},
            ],
            duration_ms=12300,
        ),
        
        # ─── Scenario 7: Authentication Failure cascade ──────────────────
        FailureScenario(
            scenario_id="S07",
            name="5G-AKA Authentication Failure",
            description="AUSF authentication failure due to SQN synchronization causes UE registration rejection",
            root_cause="UE31 5G-AKA authentication failure at AUSF due to Sequence Number (SQN) out of synchronization between UE USIM and UDM database (delta=256, threshold=32), causing MAC verification failure in authentication vector, NAS rejection with cause #21 (synch-failure) and forced re-authentication after AUTS-based resynchronization",
            affected_ues=["UE31"],
            chain=[
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:31] NAS: Registration Request received, type=initial, 5GS-TMSI=0x4A2B1C3D", "delay_ms": 0},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:31] Nausf_UEAuthentication initiated, requesting authentication vector from AUSF", "delay_ms": 50},
                {"file": "core_nw_ausf.log", "module": "AUSF", "severity": "INFO", "error_code": "", "message": "[ueId:31] Authentication request received, method=5G-AKA, supi=imsi-310260123456789", "delay_ms": 80},
                {"file": "core_nw_ausf.log", "module": "AUSF", "severity": "INFO", "error_code": "", "message": "[ueId:31] Nudm_UEAuthentication: requesting auth vector from UDM, serving-network=5G:mnc026.mcc310", "delay_ms": 100},
                {"file": "core_nw_udm.log", "module": "UDM", "severity": "INFO", "error_code": "", "message": "[ueId:31] GenerateAuthData: computing AV, SQN_HE=0x0000000001FF, K=<redacted>", "delay_ms": 120},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:31] NAS: Authentication Request sent to UE with RAND and AUTN", "delay_ms": 200},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "ERROR", "error_code": "AMF_AUTH_FAIL_094", "message": "[ueId:31] NAS: Authentication Failure received from UE, cause=synch-failure, AUTS present", "delay_ms": 500},
                {"file": "core_nw_ausf.log", "module": "AUSF", "severity": "ERROR", "error_code": "", "message": "[ueId:31] Authentication failure: SQN synchronization error, UE_SQN=0x00000000FF, HE_SQN=0x0000000001FF, delta=256 exceeds threshold=32", "delay_ms": 520},
                {"file": "core_nw_udm.log", "module": "UDM", "severity": "WARNING", "error_code": "", "message": "[ueId:31] SQN resynchronization triggered using AUTS, updating HE SQN to match UE", "delay_ms": 550},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:31] Re-authentication initiated after SQN resync, new Authentication Request sent", "delay_ms": 700},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "[ueId:31] Authentication successful after resynchronization, proceeding with Security Mode Command", "delay_ms": 1000},
            ],
            duration_ms=1000,
        ),
        
        # ─── Scenario 8: Scheduler Starvation ────────────────────────────
        FailureScenario(
            scenario_id="S08",
            name="DL Scheduler Starvation under Load",
            description="High cell load causes proportional fair scheduler to starve low-priority UEs, triggering BSR timeouts",
            root_cause="Cell-12 DL PRB utilization reaches 95% with 87 active UEs, causing proportional fair scheduler to repeatedly skip low-CQI UEs (UE9, UE16, UE27) for 800ms+ resulting in BSR timer expiry, buffer overflow at MAC layer with 2.4MB data discarded, and application-level TCP retransmissions",
            affected_ues=["UE9", "UE16", "UE27"],
            chain=[
                {"file": "ran_cell.log", "module": "SCHEDULER", "severity": "INFO", "error_code": "", "message": "Cell-12: Active UE count=87, DL PRB utilization=95%, UL PRB utilization=72%", "delay_ms": 0},
                {"file": "ran_cell.log", "module": "SCHEDULER", "severity": "WARNING", "error_code": "", "message": "Cell-12: Proportional Fair metric imbalance, top-10 UEs consuming 68% DL resources, bottom-20 UEs starved", "delay_ms": 200},
                {"file": "enodeb_rrc.log", "module": "MAC", "severity": "WARNING", "error_code": "", "message": "UE9: No DL allocation for 400ms, CQI=4, PF priority metric too low relative to high-CQI UEs", "delay_ms": 400},
                {"file": "enodeb_rrc.log", "module": "MAC", "severity": "WARNING", "error_code": "", "message": "UE16: No DL allocation for 550ms, CQI=3, buffer occupancy rising (1.2MB pending)", "delay_ms": 550},
                {"file": "enodeb_rrc.log", "module": "MAC", "severity": "ERROR", "error_code": "MAC_BSR_TIMEOUT_064", "message": "UE27: BSR timer expired (800ms), no UL grant received, CQI=2, scheduling starvation detected", "delay_ms": 800},
                {"file": "enodeb_rrc.log", "module": "MAC", "severity": "ERROR", "error_code": "MAC_BSR_TIMEOUT_064", "message": "UE9: BSR timer expired (800ms), MAC buffer overflow, 2.4MB data discarded from transmission queue", "delay_ms": 850},
                {"file": "ran_cell.log", "module": "SCHEDULER", "severity": "ERROR", "error_code": "SCHED_DL_FAIL_131", "message": "Cell-12: Scheduling failure rate=12% (target <2%), 15 UEs experiencing >500ms scheduling gaps", "delay_ms": 900},
                {"file": "cu_up_gtpu.log", "module": "PACKET_PROC", "severity": "WARNING", "error_code": "", "message": "[ueId:9,16,27] DL packet buffering at CU-UP, combined buffer=8.7MB, approaching high watermark", "delay_ms": 1000},
                {"file": "ran_cell.log", "module": "POWER_CTRL", "severity": "INFO", "error_code": "", "message": "Cell-12: Load balancing triggered, offloading 12 UEs to neighbor Cell-13 via A5 handover", "delay_ms": 1200},
            ],
            duration_ms=1200,
        ),
        
        # ─── Scenario 9: IPSec Tunnel Failure ────────────────────────────
        FailureScenario(
            scenario_id="S09",
            name="IPSec Tunnel Failure on Midhaul",
            description="IPSec IKEv2 SA expiry without timely rekeying causes midhaul encryption failure",
            root_cause="IPSec IKEv2 SA expiry on midhaul link between DU and CU (tunnel-id=47) due to rekeying failure (DPD timeout from CU side) causes encrypted data path disruption for 4.7s, affecting F1-U data plane for 18 UEs until tunnel re-establishment with new SA",
            affected_ues=["UE2", "UE6", "UE10", "UE13", "UE17"],
            chain=[
                {"file": "transport_sctp.log", "module": "IPSec", "severity": "WARNING", "error_code": "", "message": "IPSec tunnel-47 (DU<->CU midhaul): IKEv2 SA approaching expiry, rekey initiated, lifetime=86400s remaining=120s", "delay_ms": 0},
                {"file": "transport_sctp.log", "module": "IPSec", "severity": "WARNING", "error_code": "", "message": "IPSec tunnel-47: IKEv2 CREATE_CHILD_SA request sent for rekeying, awaiting response", "delay_ms": 1000},
                {"file": "transport_sctp.log", "module": "IPSec", "severity": "ERROR", "error_code": "", "message": "IPSec tunnel-47: DPD (Dead Peer Detection) timeout, peer not responding to rekey request, retry 1/3", "delay_ms": 4000},
                {"file": "transport_sctp.log", "module": "IPSec", "severity": "ERROR", "error_code": "", "message": "IPSec tunnel-47: SA EXPIRED, no valid SA for encryption, data plane disrupted for 18 tunneled flows", "delay_ms": 7000},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "ERROR", "error_code": "", "message": "F1-U data plane error: encrypted tunnel-47 unavailable, 18 UE bearers affected, packets queued", "delay_ms": 7100},
                {"file": "cu_cp_f1ap.log", "module": "F1AP", "severity": "ERROR", "error_code": "F1AP_MSG_DECODE_ERR_018", "message": "F1AP: Message decode failure on affected path, encryption mismatch after SA expiry", "delay_ms": 7200},
                {"file": "transport_sctp.log", "module": "IPSec", "severity": "INFO", "error_code": "", "message": "IPSec tunnel-47: IKEv2 re-authentication initiated, new IKE SA negotiation in progress", "delay_ms": 8000},
                {"file": "transport_sctp.log", "module": "IPSec", "severity": "INFO", "error_code": "", "message": "IPSec tunnel-47: New SA established, tunnel restored, interruption duration=4.7s", "delay_ms": 11700},
                {"file": "cu_up_gtpu.log", "module": "GTP-U", "severity": "INFO", "error_code": "", "message": "F1-U data plane restored on tunnel-47, flushing 892 queued packets, service resuming for 18 UEs", "delay_ms": 11800},
            ],
            duration_ms=11800,
        ),
        
        # ─── Scenario 10: Core Network Overload (AMF) ────────────────────
        FailureScenario(
            scenario_id="S10",
            name="AMF Overload causing Registration Delays",
            description="AMF overload from registration storm causes NAS message processing delays and timeouts",
            root_cause="AMF processing overload (CPU 94%, message queue depth 2847) caused by simultaneous registration storm from 500+ UEs after cell outage recovery, resulting in NAS timer T3510 expiry for 127 UEs, registration failures, and AMF-initiated overload control with back-off timer=30s broadcast to gNBs",
            affected_ues=["UE40", "UE41", "UE42", "UE43", "UE44"],
            chain=[
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "WARNING", "error_code": "", "message": "AMF: Message queue depth=2847 (threshold=1000), processing delay=1.2s, CPU=94%", "delay_ms": 0},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "WARNING", "error_code": "", "message": "AMF: Registration storm detected, 523 pending Registration Requests in 10s window", "delay_ms": 500},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "ERROR", "error_code": "AMF_REG_REJECT_091", "message": "AMF: Overload condition triggered, rejecting new registrations with cause=#22 (congestion), back-off=30s", "delay_ms": 1000},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "ERROR", "error_code": "", "message": "[ueId:40,41,42,43,44...] NAS: T3510 timer expiry for 127 UEs, registration procedure abandoned", "delay_ms": 1500},
                {"file": "cu_cp_f1ap.log", "module": "NGAP", "severity": "WARNING", "error_code": "", "message": "NGAP: AMF Overload Start received, reducing Initial UE Message rate, backoff=30s", "delay_ms": 1600},
                {"file": "cu_cp_f1ap.log", "module": "NGAP", "severity": "INFO", "error_code": "", "message": "NGAP: Rate limiting applied, buffering new Registration Requests at gNB, queue=89", "delay_ms": 1700},
                {"file": "ran_cell.log", "module": "CELL_MGMT", "severity": "WARNING", "error_code": "", "message": "Cell-12,13,14: AMF overload notification received, deferring non-emergency registrations", "delay_ms": 1800},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "AMF: Overload mitigation active, processing queue draining, depth=1200, delay=0.4s", "delay_ms": 5000},
                {"file": "core_nw_amf.log", "module": "AMF", "severity": "INFO", "error_code": "", "message": "AMF: Overload resolved, CPU=62%, queue depth=89, resuming normal registration processing", "delay_ms": 10000},
                {"file": "cu_cp_f1ap.log", "module": "NGAP", "severity": "INFO", "error_code": "", "message": "NGAP: AMF Overload Stop received, releasing rate limiting, processing buffered requests", "delay_ms": 10100},
            ],
            duration_ms=10100,
        ),
    ]
    
    return scenarios


# ─── Log Generation ──────────────────────────────────────────────────────────

def generate_background_logs(base_time: datetime.datetime, file_name: str, 
                             module_list: List[str], count: int) -> List[Dict]:
    """Generate background/normal operation log entries."""
    logs = []
    normal_messages = {
        "RRC": [
            "UE{ue} RRC Connection Setup Complete, cellId={cell}",
            "UE{ue} Measurement Report received, serving RSRP={rsrp}dBm",
            "UE{ue} RRC Release sent, cause=other",
            "UE{ue} Security Mode Complete received",
            "UE{ue} RRC Connected, rnti=0x{rnti:04X}",
        ],
        "PDCP": [
            "UE{ue} PDCP SDU delivery rate=99.97%, bearer={drb}",
            "UE{ue} PDCP SN=0x{sn:04X}, integrity verification passed",
            "UE{ue} PDCP stats: txPDU={tx}, rxPDU={rx}, discarded=0",
        ],
        "RLC": [
            "UE{ue} RLC stats: txPDU={tx}, rxPDU={rx}, retx={retx}",
            "UE{ue} RLC AM mode active on DRB-{drb}, polling timer=45ms",
        ],
        "MAC": [
            "UE{ue} MAC: DL throughput={tput}Mbps, CQI={cqi}, rank={rank}",
            "UE{ue} MAC: RACH procedure completed, preamble={preamble}, TA={ta}",
            "UE{ue} MAC: SR transmitted, BSR={bsr}bytes pending",
        ],
        "F1AP": [
            "[ueId:{ue}] UL RRC Message Transfer received, lcId=1, msgSize={size}bytes",
            "[ueId:{ue}] DL RRC Message Transfer sent successfully",
            "[ueId:{ue}] F1AP: UE Context Setup Response sent",
        ],
        "NGAP": [
            "NGAP: Initial UE Message sent to AMF for ueId:{ue}",
            "NGAP: Downlink NAS Transport for ueId:{ue}, msgSize={size}bytes",
            "NGAP: Heartbeat to AMF successful, RTT=2ms",
        ],
        "GTP-U": [
            "[ueId:{ue}] GTP-U: DL packets forwarded, count={count}, bytes={bytes}",
            "[ueId:{ue}] GTP-U: tunnel teid=0x{teid:08X} active, throughput={tput}Mbps",
            "GTP-U: Echo Response received from UPF (10.0.2.50), RTT=2ms",
        ],
        "AMF": [
            "[ueId:{ue}] NAS: Registration Accept sent, 5G-GUTI assigned",
            "[ueId:{ue}] NAS: Service Accept, pduSessionStatus=active",
            "AMF: System health OK, CPU=42%, memory=67%, msg_queue=124",
        ],
        "SMF": [
            "[ueId:{ue}] PDU Session active, sessionId={sess}, upf=10.0.2.50",
            "[ueId:{ue}] QoS flow established, 5QI={qos}, AMBR_DL={ambr}Mbps",
        ],
        "SCTP": [
            "SCTP association id={assoc}: heartbeat OK, RTT=1ms, state=ESTABLISHED",
            "SCTP association id={assoc}: stream {stream} active, chunks_sent={chunks}",
        ],
        "CELL_MGMT": [
            "Cell-{cell}: Operational state=ENABLED, PRB_DL={prb}%, UE_count={ue_count}",
            "Cell-{cell}: KPI report: avg_throughput_DL={tput}Mbps, avg_latency={lat}ms",
        ],
        "SCHEDULER": [
            "Cell-{cell}: DL scheduled {n_ues} UEs, total PRBs={prbs}, efficiency={eff}%",
            "Cell-{cell}: UL grants issued={grants}, SR pending={sr}",
        ],
        "RADIO_RESOURCE": [
            "Cell-{cell}: Admission control: {admitted} admitted, {rejected} rejected this interval",
            "Cell-{cell}: Average RSRP={rsrp}dBm across {n} connected UEs",
        ],
        "BACKHAUL": [
            "Backhaul link eth2: status=UP, throughput={tput}Gbps, latency={lat}ms, BER<1e-12",
            "Backhaul link eth3: standby, last failover test=OK",
        ],
        "IPSec": [
            "IPSec tunnel-{tid}: SA valid, lifetime remaining={life}s, packets encrypted={pkts}",
        ],
        "BEARER_MGMT": [
            "[ueId:{ue}] Active bearers: DRB-1 (5QI=9), DRB-2 (5QI=7), status=normal",
        ],
        "UE_CONTEXT": [
            "[ueId:{ue}] UE Context active, state=CONNECTED, inactivity_timer=60s",
        ],
        "HANDOVER": [
            "[ueId:{ue}] Handover evaluation: current RSRP={rsrp}dBm, threshold not met, staying",
        ],
        "PACKET_PROC": [
            "[ueId:{ue}] PacketProcessor: DL throughput={tput}Mbps, jitter={jitter}ms, loss=0%",
        ],
        "QOS": [
            "[ueId:{ue}] QoS enforcement: flow {flow} within AMBR limits, marking=DSCP-46",
        ],
        "POWER_CTRL": [
            "Cell-{cell}: DL power control adjusted, Ptx={ptx}dBm, coverage radius={radius}m",
        ],
        "INTERFERENCE": [
            "Cell-{cell}: Inter-cell interference level={level}dB, within acceptable range",
        ],
        "PAGING": [
            "Paging: UE paged in TAI-list, 5G-S-TMSI=0x{tmsi:08X}, attempt=1",
        ],
        "FRONTHAUL": [
            "Fronthaul link eCPRI-{port}: sync status=locked, delay_asymmetry<50ns",
        ],
        "SDAP": [
            "[ueId:{ue}] SDAP: QFI={qfi} mapped to DRB-{drb}, reflective QoS=disabled",
        ],
    }
    
    for i in range(count):
        module = random.choice(module_list)
        time_offset = random.uniform(0, 60000)  # within 60s window
        ts = base_time + datetime.timedelta(milliseconds=time_offset)
        
        templates = normal_messages.get(module, [f"Module {module}: normal operation, status=OK"])
        template = random.choice(templates)
        
        # Fill template
        msg = template.format(
            ue=random.randint(1, 50),
            cell=random.randint(12, 16),
            rsrp=random.randint(-110, -70),
            rnti=random.randint(0x1000, 0xFFFF),
            sn=random.randint(0, 0xFFFF),
            drb=random.randint(1, 3),
            tx=random.randint(1000, 50000),
            rx=random.randint(1000, 50000),
            retx=random.randint(0, 50),
            tput=round(random.uniform(10, 500), 1),
            cqi=random.randint(5, 15),
            rank=random.randint(1, 4),
            preamble=random.randint(0, 63),
            ta=random.randint(0, 1282),
            bsr=random.randint(100, 50000),
            size=random.randint(10, 200),
            count=random.randint(100, 10000),
            bytes=random.randint(10000, 5000000),
            teid=random.randint(0x10000, 0xFFFFF),
            sess=random.randint(1, 5),
            qos=random.choice([1, 2, 5, 7, 9]),
            ambr=random.choice([50, 100, 200, 500]),
            assoc=random.randint(1, 5),
            stream=random.randint(0, 10),
            chunks=random.randint(100, 10000),
            prb=random.randint(30, 75),
            ue_count=random.randint(20, 80),
            lat=round(random.uniform(1, 10), 1),
            n_ues=random.randint(10, 60),
            prbs=random.randint(50, 270),
            eff=random.randint(75, 98),
            grants=random.randint(10, 50),
            sr=random.randint(0, 10),
            admitted=random.randint(0, 5),
            rejected=random.randint(0, 2),
            n=random.randint(20, 80),
            tid=random.randint(40, 55),
            life=random.randint(1000, 86000),
            pkts=random.randint(10000, 1000000),
            tmsi=random.randint(0x10000000, 0xFFFFFFFF),
            flow=random.randint(1, 4),
            ptx=random.randint(30, 46),
            radius=random.randint(200, 1500),
            level=round(random.uniform(-5, 5), 1),
            port=random.randint(1, 4),
            qfi=random.randint(1, 9),
            jitter=round(random.uniform(0.1, 5), 1),
        )
        
        severity = random.choices(
            ["DEBUG", "INFO", "INFO", "INFO", "WARNING"],
            weights=[0.05, 0.5, 0.3, 0.1, 0.05]
        )[0]
        
        logs.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "severity": severity,
            "module": module,
            "error_code": "",
            "message": msg,
            "file": file_name,
        })
    
    return logs


def generate_scenario_logs(scenario: FailureScenario, base_time: datetime.datetime) -> List[Dict]:
    """Generate log entries for a specific failure scenario."""
    logs = []
    for step in scenario.chain:
        ts = base_time + datetime.timedelta(milliseconds=step["delay_ms"])
        logs.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "severity": step["severity"],
            "module": step["module"],
            "error_code": step.get("error_code", ""),
            "message": step["message"],
            "file": step["file"],
        })
    return logs


def format_log_line(entry: Dict) -> str:
    """Format a single log entry as a text line."""
    ec = f" [{entry['error_code']}]" if entry["error_code"] else ""
    return f"{entry['timestamp']} [{entry['severity']:8s}] [{entry['module']:16s}]{ec} {entry['message']}"


def generate_full_dataset():
    """Generate the complete synthetic dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    scenarios = build_scenarios()
    base_time = datetime.datetime(2025, 3, 15, 14, 30, 0)
    
    # Define log files and their associated modules
    log_files = {
        "enodeb_rrc.log": MODULES["enodeb"],
        "cu_cp_f1ap.log": MODULES["cu_cp"],
        "cu_up_gtpu.log": MODULES["cu_up"],
        "ran_cell.log": MODULES["ran"],
        "core_nw_amf.log": MODULES["core_nw"][:3],  # AMF, SMF, UPF
        "core_nw_smf.log": ["SMF"],
        "core_nw_pcf.log": ["PCF"],
        "core_nw_ausf.log": ["AUSF"],
        "core_nw_udm.log": ["UDM"],
        "transport_sctp.log": MODULES["transport"],
    }
    
    # Generate background logs for each file (normal operations)
    all_logs_by_file = {f: [] for f in log_files}
    
    for file_name, modules in log_files.items():
        bg_count = random.randint(60, 120)
        bg_logs = generate_background_logs(base_time, file_name, modules, bg_count)
        all_logs_by_file[file_name].extend(bg_logs)
    
    # Inject scenario events at different time offsets
    scenario_offsets = [0, 15000, 30000, 45000, 60000, 90000, 120000, 150000, 180000, 210000]
    
    for scenario, offset_ms in zip(scenarios, scenario_offsets):
        scenario_base = base_time + datetime.timedelta(milliseconds=offset_ms)
        scenario_logs = generate_scenario_logs(scenario, scenario_base)
        
        for log_entry in scenario_logs:
            file_name = log_entry["file"]
            if file_name in all_logs_by_file:
                all_logs_by_file[file_name].append(log_entry)
    
    # Sort each file's logs by timestamp and write
    total_entries = 0
    for file_name, logs in all_logs_by_file.items():
        logs.sort(key=lambda x: x["timestamp"])
        
        output_path = OUTPUT_DIR / file_name
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# {file_name} — Synthetic 5G Telecom Log\n")
            f.write(f"# Generated for RAG evaluation — {len(logs)} entries\n")
            f.write(f"# Time range: {base_time.isoformat()} + 240s\n")
            f.write("#" + "=" * 79 + "\n\n")
            
            for entry in logs:
                f.write(format_log_line(entry) + "\n")
        
        total_entries += len(logs)
        print(f"  {file_name}: {len(logs)} entries")
    
    print(f"\nTotal: {total_entries} log entries across {len(log_files)} files")
    return scenarios, total_entries


# ─── Query Generation ────────────────────────────────────────────────────────

def generate_queries(scenarios: List[FailureScenario]) -> List[Dict]:
    """Generate 50+ evaluation queries with ground truth."""
    
    queries = []
    query_id = 1
    
    for scenario in scenarios:
        # Generate multiple query types per scenario
        
        # Type 1: Root cause analysis (broad)
        queries.append({
            "id": f"Q{query_id:03d}",
            "query": f"What is the root cause of the failure affecting {scenario.affected_ues[0]}?",
            "type": "root_cause",
            "difficulty": "medium",
            "scenario_id": scenario.scenario_id,
            "ground_truth": {
                "root_cause": scenario.root_cause,
                "relevant_files": list(set(step["file"] for step in scenario.chain)),
                "relevant_modules": list(set(step["module"] for step in scenario.chain)),
                "error_codes": [step["error_code"] for step in scenario.chain if step["error_code"]],
                "affected_ues": scenario.affected_ues,
                "requires_multi_hop": len(set(step["file"] for step in scenario.chain)) > 1,
            },
            "keywords": _extract_keywords(scenario),
        })
        query_id += 1
        
        # Type 2: Failure tracing (specific)
        error_steps = [s for s in scenario.chain if s["severity"] in ("ERROR", "CRITICAL")]
        if error_steps:
            first_error = error_steps[0]
            queries.append({
                "id": f"Q{query_id:03d}",
                "query": f"What error occurred in the {first_error['module']} module and what were its downstream effects?",
                "type": "failure_tracing",
                "difficulty": "medium",
                "scenario_id": scenario.scenario_id,
                "ground_truth": {
                    "root_cause": scenario.root_cause,
                    "relevant_files": list(set(step["file"] for step in scenario.chain)),
                    "relevant_modules": list(set(step["module"] for step in scenario.chain)),
                    "error_codes": [step["error_code"] for step in scenario.chain if step["error_code"]],
                    "affected_ues": scenario.affected_ues,
                    "requires_multi_hop": True,
                },
                "keywords": _extract_keywords(scenario),
            })
            query_id += 1
        
        # Type 3: Multi-hop reasoning
        files_involved = list(set(step["file"] for step in scenario.chain))
        if len(files_involved) > 1:
            queries.append({
                "id": f"Q{query_id:03d}",
                "query": f"Trace the complete chain of events from {scenario.chain[0]['file']} to {scenario.chain[-1]['file']} that led to the {scenario.name.lower()}.",
                "type": "multi_hop",
                "difficulty": "hard",
                "scenario_id": scenario.scenario_id,
                "ground_truth": {
                    "root_cause": scenario.root_cause,
                    "relevant_files": files_involved,
                    "relevant_modules": list(set(step["module"] for step in scenario.chain)),
                    "error_codes": [step["error_code"] for step in scenario.chain if step["error_code"]],
                    "affected_ues": scenario.affected_ues,
                    "requires_multi_hop": True,
                },
                "keywords": _extract_keywords(scenario),
            })
            query_id += 1
        
        # Type 4: Time-anchored
        first_ts = scenario.chain[0]["delay_ms"]
        queries.append({
            "id": f"Q{query_id:03d}",
            "query": f"What events occurred during the {scenario.name.lower()} incident and in what order?",
            "type": "temporal",
            "difficulty": "medium",
            "scenario_id": scenario.scenario_id,
            "ground_truth": {
                "root_cause": scenario.root_cause,
                "relevant_files": list(set(step["file"] for step in scenario.chain)),
                "relevant_modules": list(set(step["module"] for step in scenario.chain)),
                "error_codes": [step["error_code"] for step in scenario.chain if step["error_code"]],
                "affected_ues": scenario.affected_ues,
                "requires_multi_hop": True,
            },
            "keywords": _extract_keywords(scenario),
        })
        query_id += 1
        
        # Type 5: Impact analysis
        queries.append({
            "id": f"Q{query_id:03d}",
            "query": f"What was the impact of the {scenario.name.lower()} on user equipment and network services?",
            "type": "impact_analysis",
            "difficulty": "medium",
            "scenario_id": scenario.scenario_id,
            "ground_truth": {
                "root_cause": scenario.root_cause,
                "relevant_files": list(set(step["file"] for step in scenario.chain)),
                "relevant_modules": list(set(step["module"] for step in scenario.chain)),
                "error_codes": [step["error_code"] for step in scenario.chain if step["error_code"]],
                "affected_ues": scenario.affected_ues,
                "requires_multi_hop": True,
            },
            "keywords": _extract_keywords(scenario),
        })
        query_id += 1
        
        # Type 6: Error code specific
        error_codes = [s["error_code"] for s in scenario.chain if s["error_code"]]
        if error_codes:
            queries.append({
                "id": f"Q{query_id:03d}",
                "query": f"What does error code {error_codes[0]} indicate and what caused it?",
                "type": "error_code",
                "difficulty": "easy",
                "scenario_id": scenario.scenario_id,
                "ground_truth": {
                    "root_cause": scenario.root_cause,
                    "relevant_files": list(set(step["file"] for step in scenario.chain if step["error_code"] == error_codes[0])),
                    "relevant_modules": list(set(step["module"] for step in scenario.chain)),
                    "error_codes": error_codes,
                    "affected_ues": scenario.affected_ues,
                    "requires_multi_hop": False,
                },
                "keywords": _extract_keywords(scenario),
            })
            query_id += 1
    
    # Add cross-scenario queries
    cross_queries = [
        {
            "id": f"Q{query_id:03d}",
            "query": "Which failures affected multiple UEs simultaneously and what was the common cause?",
            "type": "cross_scenario",
            "difficulty": "hard",
            "scenario_id": "S02,S06,S08,S10",
            "ground_truth": {
                "root_cause": "SCTP transport failure (S02), GTP-U path failure (S06), scheduler starvation (S08), and AMF overload (S10) all affected multiple UEs. Common pattern: infrastructure-level failures cascade to multiple UE contexts.",
                "relevant_files": ["transport_sctp.log", "cu_cp_f1ap.log", "cu_up_gtpu.log", "core_nw_amf.log", "ran_cell.log"],
                "relevant_modules": ["SCTP", "F1AP", "GTP-U", "AMF", "SCHEDULER"],
                "error_codes": ["SCTP_ASSOC_FAIL_081", "GTPU_PATH_FAIL_037", "SCHED_DL_FAIL_131", "AMF_REG_REJECT_091"],
                "affected_ues": ["UE1-UE5", "multiple"],
                "requires_multi_hop": True,
            },
            "keywords": ["sctp", "gtp-u", "path failure", "overload", "multiple ues", "transport"],
        },
    ]
    query_id += 1
    
    cross_queries.append({
        "id": f"Q{query_id:03d}",
        "query": "Compare the transport-layer failures: how does SCTP association failure differ from GTP-U path failure in terms of impact?",
        "type": "cross_scenario",
        "difficulty": "hard",
        "scenario_id": "S02,S06",
        "ground_truth": {
            "root_cause": "SCTP failure (S02) affects control plane (F1AP signaling) causing immediate UE context release for all UEs on that link (45 UEs). GTP-U path failure (S06) affects user plane (data forwarding) causing data buffering/loss for active data sessions (28 tunnels) but control plane remains intact allowing recovery via path switch.",
            "relevant_files": ["transport_sctp.log", "cu_cp_f1ap.log", "cu_up_gtpu.log", "core_nw_amf.log"],
            "relevant_modules": ["SCTP", "GTP-U", "F1AP", "BEARER_MGMT"],
            "error_codes": ["SCTP_ASSOC_FAIL_081", "GTPU_PATH_FAIL_037"],
            "affected_ues": ["UE1-UE5", "UE3,UE8,UE14,UE19,UE25"],
            "requires_multi_hop": True,
        },
        "keywords": ["sctp", "gtp-u", "control plane", "user plane", "impact", "45 ues", "28 tunnels"],
    })
    query_id += 1
    
    cross_queries.append({
        "id": f"Q{query_id:03d}",
        "query": "What are all the scenarios where packet loss or data loss occurred and what were the root causes?",
        "type": "cross_scenario",
        "difficulty": "hard",
        "scenario_id": "S01,S03,S06",
        "ground_truth": {
            "root_cause": "Packet/data loss occurred in: (1) S01 - 23 packets dropped due to GTP-U tunnel teardown after RRC reconfig failure; (2) S03 - 847 PDCP SDUs lost during RLF from failed handover; (3) S06 - buffered packets during 3.2s GTP-U path interruption. Common pattern: control plane failures propagate to user plane data loss.",
            "relevant_files": ["cu_up_gtpu.log", "enodeb_rrc.log", "cu_cp_f1ap.log"],
            "relevant_modules": ["GTP-U", "PACKET_PROC", "PDCP", "RRC"],
            "error_codes": ["GTPU_TUNNEL_ERR_031", "PDCP_SN_OVERFLOW_047", "GTPU_PATH_FAIL_037"],
            "affected_ues": ["UE7", "UE11", "UE3,UE8,UE14,UE19,UE25"],
            "requires_multi_hop": True,
        },
        "keywords": ["packet loss", "data loss", "forward jump", "pdcp", "gtp-u", "tunnel"],
    })
    query_id += 1
    
    cross_queries.append({
        "id": f"Q{query_id:03d}",
        "query": "Did any UE recover successfully after a failure, and if so, how long did the service interruption last?",
        "type": "cross_scenario",
        "difficulty": "medium",
        "scenario_id": "S03,S06,S07,S09",
        "ground_truth": {
            "root_cause": "Successful recoveries: (1) UE11 (S03) - RRC re-establishment after RLF, interruption=2.3s; (2) UEs in S06 - service resumed after backhaul failover, interruption=3.2s; (3) UE31 (S07) - re-authentication after SQN resync, delay~1s; (4) UEs in S09 - tunnel re-established after IPSec SA renegotiation, interruption=4.7s.",
            "relevant_files": ["enodeb_rrc.log", "cu_up_gtpu.log", "core_nw_amf.log", "transport_sctp.log"],
            "relevant_modules": ["RRC", "GTP-U", "AMF", "AUSF", "IPSec"],
            "error_codes": [],
            "affected_ues": ["UE11", "UE31", "multiple"],
            "requires_multi_hop": True,
        },
        "keywords": ["recovery", "re-establishment", "interruption", "restored", "resumed"],
    })
    query_id += 1
    
    # Contrastive queries
    cross_queries.append({
        "id": f"Q{query_id:03d}",
        "query": "Was the UE7 failure related to the SCTP transport failure, or were they independent events?",
        "type": "contrastive",
        "difficulty": "medium",
        "scenario_id": "S01,S02",
        "ground_truth": {
            "root_cause": "Independent events. UE7 failure (S01) was caused by RRC Reconfiguration failure code 4 due to invalid measurement config - a radio-layer issue specific to UE7. SCTP transport failure (S02) was a transport-layer issue (heartbeat timeout) affecting all 45 UEs on that SCTP association. Different root causes, different affected UEs, different network layers.",
            "relevant_files": ["enodeb_rrc.log", "transport_sctp.log", "cu_cp_f1ap.log"],
            "relevant_modules": ["RRC", "SCTP", "F1AP", "UE_CONTEXT"],
            "error_codes": ["RRC_RECONFIG_FAIL_004", "SCTP_ASSOC_FAIL_081"],
            "affected_ues": ["UE7", "UE1-UE5"],
            "requires_multi_hop": True,
        },
        "keywords": ["independent", "rrc", "sctp", "transport", "radio", "ue7"],
    })
    query_id += 1
    
    queries.extend(cross_queries)
    
    return queries


def _extract_keywords(scenario: FailureScenario) -> List[str]:
    """Extract evaluation keywords from a scenario."""
    keywords = []
    # Add error codes
    for step in scenario.chain:
        if step["error_code"]:
            keywords.append(step["error_code"].lower())
    # Add key terms from root cause
    root_lower = scenario.root_cause.lower()
    for term in ["failure", "timeout", "error", "loss", "reject", "expired", "overflow"]:
        if term in root_lower:
            keywords.append(term)
    # Add affected UEs
    keywords.extend([ue.lower() for ue in scenario.affected_ues[:3]])
    # Add key modules
    modules = list(set(step["module"].lower() for step in scenario.chain if step["severity"] in ("ERROR", "CRITICAL")))
    keywords.extend(modules[:4])
    return list(set(keywords))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Generating Synthetic Telecom Log Dataset for RAG Evaluation")
    print("=" * 70)
    print()
    
    print("Generating log files...")
    scenarios, total_entries = generate_full_dataset()
    
    print(f"\nGenerating evaluation queries...")
    queries = generate_queries(scenarios)
    
    # Save queries
    dataset = {
        "metadata": {
            "description": "Synthetic telecom log evaluation dataset for RAG systems",
            "num_scenarios": len(scenarios),
            "num_log_entries": total_entries,
            "num_queries": len(queries),
            "query_types": list(set(q["type"] for q in queries)),
            "scenarios": [{"id": s.scenario_id, "name": s.name, "description": s.description} for s in scenarios],
        },
        "queries": queries,
    }
    
    with open(QUERIES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"  Generated {len(queries)} queries across {len(set(q['type'] for q in queries))} types")
    print(f"  Saved to: {QUERIES_OUTPUT}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"  Log files:     {len([f for f in OUTPUT_DIR.iterdir() if f.suffix == '.log'])}")
    print(f"  Total entries: {total_entries}")
    print(f"  Scenarios:     {len(scenarios)}")
    print(f"  Queries:       {len(queries)}")
    print(f"\n  Query type breakdown:")
    from collections import Counter
    type_counts = Counter(q["type"] for q in queries)
    for qtype, count in sorted(type_counts.items()):
        print(f"    {qtype:20s}: {count}")
    
    print(f"\n  Difficulty breakdown:")
    diff_counts = Counter(q["difficulty"] for q in queries)
    for diff, count in sorted(diff_counts.items()):
        print(f"    {diff:10s}: {count}")
    
    print(f"\n  Multi-hop queries: {sum(1 for q in queries if q['ground_truth'].get('requires_multi_hop', False))}")
    print(f"  Cross-file queries: {sum(1 for q in queries if len(q['ground_truth'].get('relevant_files', [])) > 1)}")


if __name__ == "__main__":
    main()
