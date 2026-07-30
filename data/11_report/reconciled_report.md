# Reconciled Compliance Assessment — Human key vs Claude judge

> Both the human answer key and the Claude judge were rolled up to per-control status. Where they differ, the judge's evidence is shown for adjudication. The judge finds substantial coverage the human key recorded as gaps.

## Summary (assessed controls)

| Status | Human key | Claude judge |
|---|---|---|
| Met | 8 | 8 |
| Partially Met | 7 | 44 |
| Gap | 66 | 29 |

- Controls where the verdict changed: **51** of 81
- Controls the human marked **Gap** but the judge finds **covered**: **43**
- Controls the human marked covered but the judge finds **Gap**: **6** (check these — likely mis-pooled passages or human over-labels)
- Pair-level disagreements to adjudicate: **70**

## Controls the human key under-counted (Gap -> Covered)

These were checked and marked a gap, but a policy passage does address them:

- **M2.1.1** Unknown — human: Gap - judge: **Partially Met**
- **M3.2.1** Awareness and Training — human: Gap - judge: **Partially Met**
- **M5.2.1** Identification of Applicable — human: Gap - judge: **Partially Met**
- **M5.2.2** Intellectual Property — human: Gap - judge: **Partially Met**
- **M5.2.3** Protection of — human: Gap - judge: **Partially Met**
- **M5.4.1** Technical Compliance — human: Gap - judge: **Partially Met**
- **T1.2.3**  — human: Gap - judge: **Partially Met**
- **T1.3.2** Labeling of Information — human: Gap - judge: **Partially Met**
- **T1.4.1** Management of — human: Gap - judge: **Partially Met**
- **T1.4.2** Disposal of Media — human: Gap - judge: **Met**
- **T2.2.2** Physical Entry Controls — human: Gap - judge: **Met**
- **T2.2.4**  — human: Gap - judge: **Partially Met**
- **T2.2.5** Working in Secure Areas — human: Gap - judge: **Partially Met**
- **T2.2.6**  — human: Gap - judge: **Partially Met**
- **T2.3.3**  — human: Gap - judge: **Partially Met**
- **T2.3.5**  — human: Gap - judge: **Partially Met**
- **T2.3.6** Secure Disposal or Re-Use — human: Gap - judge: **Partially Met**
- **T3.2.2** Documented Operating — human: Gap - judge: **Met**
- **T3.2.4**  — human: Gap - judge: **Partially Met**
- **T3.2.5** Separation of Development, — human: Gap - judge: **Partially Met**
- **T3.3.1** Capacity Management — human: Gap - judge: **Met**
- **T4.2.1** Information Transfer — human: Gap - judge: **Met**
- **T4.2.2** Agreements on — human: Gap - judge: **Partially Met**
- **T4.2.3** Physical Media in Transit — human: Gap - judge: **Met**
- **T4.2.4**  — human: Gap - judge: **Partially Met**
- **T4.4.2**  — human: Gap - judge: **Partially Met**
- **T4.5.1** Network Controls — human: Gap - judge: **Partially Met**
- **T4.5.2** Security of Network — human: Gap - judge: **Partially Met**
- **T4.5.3**  — human: Gap - judge: **Partially Met**
- **T4.5.4** Security of Wireless — human: Gap - judge: **Partially Met**
- **T5.2.3** User Security Credentials — human: Gap - judge: **Partially Met**
- **T5.2.4** Review of User Access — human: Gap - judge: **Partially Met**
- **T5.4.1** Policy on Use of Network — human: Gap - judge: **Partially Met**
- **T5.4.7**  — human: Gap - judge: **Partially Met**
- **T5.5.1** Secure Log-On Procedures — human: Gap - judge: **Partially Met**
- **T5.6.3** Publicly Accessible — human: Gap - judge: **Met**
- **T6.2.2**  — human: Gap - judge: **Partially Met**
- **T7.7.1** Control of Technical — human: Gap - judge: **Partially Met**
- **T8.2.2** Computer Security Incident — human: Gap - judge: **Partially Met**
- **T8.2.3** Incident Classification — human: Gap - judge: **Partially Met**
- **T8.2.5**  — human: Gap - judge: **Partially Met**
- **T8.2.6** Incident Response — human: Gap - judge: **Partially Met**
- **T8.2.8** Learning From Information — human: Gap - judge: **Partially Met**

## Controls to double-check (human covered -> judge Gap)

- **M3.3.4** Training Results — human: Met - judge: **Gap**
- **T2.1.1** Physical and Environmental — human: Met - judge: **Gap**
- **T3.6.3** Monitoring System Use — human: Met - judge: **Gap**
- **T4.1.1** Communications Policy — human: Met - judge: **Gap**
- **T6.2.3** Managing Changes to Third — human: Partially Met - judge: **Gap**
- **T8.3.2** Reporting Information — human: Met - judge: **Gap**

## Adjudication list (pair-level disagreements)

Resolve each: if the judge is right, the answer key was wrong and should be corrected.

| Control | Passage | Human | Judge | Evidence (judge) |
|---|---|---|---|---|
| M2.1.1 | Information Risk Management Policy p7 | Not  | **Partially** | 5.1.4 establish and implement a plan for communicating risk information and consulting key stakeholders during |
| M2.1.1 | Information Risk Management Policy p7 | Not  | **Partially** | 5.1.4 plan for communicating risk information and consulting key stakeholders during all stages |
| M2.1.1 | Information Risk Management Policy p7 | Not  | **Partially** | 5.1.4 plan for communicating risk information and consulting key stakeholders during all stages |
| M3.2.1 | Security Awareness and Training Po p7 | Not  | **Partially** | 5.1.1 develop and formalize awareness/training program; 5.1.6-5.1.7 provide training; 5.1.4 review/update on r |
| M3.3.4 | Security Awareness and Training Po p9 | Full | **Not Addre** | training RECORDS retention; not measuring training effectiveness before/after |
| M5.2.1 | Information Security Compliance Po p7 | Not  | **Partially** | 5.1 Identify/define/establish compliance requirements; define+implement controls, assign responsibilities |
| M5.2.2 | Information Security Compliance Po p8 | Not  | **Partially** | 5.2 IPR: identify IPR assets, system requirements, controls (reputable sources, license verification, awarenes |
| M5.2.3 | Information Security Compliance Po p9 | Not  | **Partially** | 5.3 define/categorize records + retention; controls to protect records from loss/deterioration |
| M5.4.1 | Vulnerability Management Policy v1 p7 | Not  | **Partially** | 6.1 routine VA scans (annual/quarterly), reviewed by IS team, remediation reporting |
| T1.2.3 | Asset Management Policy 6 p14 | Not  | **Partially** | 5.8 Acceptable Use of Assets: approval before deployment, documented acceptable-use rules |
| T1.2.3 | Asset Management Policy p14 | Not  | **Partially** | 5.8 Acceptable Use of Assets: rules established and circulated |
| T1.3.2 | Asset Management Policy 6 p9 | Not  | **Partially** | 5.3 labeling procedures for physical/electronic assets + apply labels on outputs |
| T1.3.2 | Asset Management Policy p9 | Not  | **Partially** | 5.3 Asset Labelling: develop labeling procedures + apply labels on outputs |
| T1.3.3 | Asset Management Policy 6 p10 | Not  | **Partially** | 5.4 develop handling procedures for processing/storing/communicating + safeguard per procedures |
| T1.3.3 | Asset Management Policy 6 p9 | Part | **Not Addre** | labelling passage; not asset-handling procedures |
| T1.3.3 | Asset Management Policy p10 | Not  | **Partially** | 5.4 Asset Handling: handling procedures + safeguard |
| T1.4.1 | Asset Management Policy 6 p11 | Not  | **Partially** | 5.5 Removable Media Management: lifecycle handling + restrict/monitor usage |
| T1.4.1 | Asset Management Policy p11 | Not  | **Partially** | 5.5 Removable Media Management: lifecycle handling + restrict/monitor usage |
| T1.4.2 | Asset Management Policy 6 p12 | Not  | **Fully Add** | 5.6 Disposal: secure disposal procedures by sensitivity, destroy paper+digital, keep records when no longer ne |
| T1.4.2 | Asset Management Policy p12 | Not  | **Fully Add** | 5.6 Disposal: secure disposal by sensitivity, destroy paper+digital, keep records |
| T2.1.1 | Physical and Environmental Securit p4 | Full | **Not Addre** | purpose fragment; does not establish full policy (roles/ack/review absent) - human FA over-labels |
| T2.2.2 | Physical and Environmental Securit p8 | Not  | **Fully Add** | 5.2 authenticate persons (5.2.8), log+monitor access (5.2.9), visible ID (5.2.6), escort contractors (5.2.12) |
| T2.2.4 | Physical and Environmental Securit p9 | Not  | **Partially** | 5.3 Protecting Against External and Environmental Threats implements the control |
| T2.2.5 | Physical and Environmental Securit p10 | Not  | **Partially** | 5.4 establish guidelines for working in secure areas + personnel aware/accept |
| T2.2.6 | Physical and Environmental Securit p11 | Not  | **Partially** | 5.5 Protect Delivery and Loading Areas implements public-access/delivery control |
| T2.3.3 | Physical and Environmental Securit p13 | Not  | **Partially** | 5.7 Cabling Security implements the control |
| T2.3.5 | Physical and Environmental Securit p14 | Not  | **Partially** | 5.8 Security of Equipment Off-Premises implements the control |
| T2.3.6 | Physical and Environmental Securit p15 | Not  | **Partially** | 5.9 destroy/overwrite media with confidential info before disposal/reuse (no records atom) |
| T3.2.2 | Security Operations Policy p6 | Not  | **Fully Add** | 5.1 operating procedures documented+approved, reviewed periodically, kept up-to-date and available to users |
| T3.2.4 | Security Operations Policy p11 | Not  | **Partially** | 5.6 Segregation of Duties implements the control |
| T3.2.5 | Security Operations Policy p8 | Not  | **Partially** | 5.3 production separated from test/dev, documented transfer procedures via SDLC procedure |
| T3.3.1 | Security Operations Policy p9 | Not  | **Fully Add** | 5.4.3-5.4.4 monitor capacity, formal capacity management process, capacity plans to meet projected demand |
| T3.6.3 | Vulnerability Management Policy v1 p5 | Full | **Not Addre** | scope boilerplate only; no monitoring types/frequency/review - human FA over-labels |
| T4.1.1 | Network and Communications Securit p4 | Full | **Not Addre** | purpose fragment; does not establish full policy - human FA over-labels |
| T4.2.1 | Network and Communications Securit p11 | Not  | **Fully Add** | 5.5 establish transfer procedures; conditions, controls, actions on issues |
| T4.2.2 | Network and Communications Securit p12 | Not  | **Partially** | 5.6 establish exchange agreements with security conditions and responsibilities |
| T4.2.3 | Asset Management Policy 6 p13 | Not  | **Fully Add** | 5.7 labelling requirements, tracking, transfer logs, loss measures for physical media in transit |
| T4.2.3 | Asset Management Policy p13 | Not  | **Fully Add** | 5.7 labelling, tracking, transfer logs, loss measures |
| T4.2.4 | Network and Communications Securit p13 | Not  | **Partially** | 5.7 Electronic Messaging implements the control |
| T4.4.2 | Network and Communications Securit p15 | Not  | **Partially** | 5.9 Information Released into Information Sharing Communities implements the control |
| T4.5.1 | Network and Communications Securit p7 | Not  | **Partially** | 5.1 network design w/ risk assessment, specific network controls, logging/monitoring |
| T4.5.2 | Network and Communications Securit p8 | Not  | **Partially** | 5.2 identify+implement security features/service levels; right-to-audit in agreements |
| T4.5.3 | Network and Communications Securit p9 | Not  | **Partially** | 5.3 Segregation in Networks implements the control |
| T4.5.4 | Network and Communications Securit p10 | Not  | **Partially** | 5.4 wireless risk assessment, segregation, controls, periodic effectiveness review |
| T5.2.2 | Access Control Policy - CPX policy p12 | Full | **Partially** | 8.6 record of privileges, separate admin ID; 2FA/logging atoms not shown - human FA over-labels |
| T5.2.3 | Access Control Policy - CPX policy p14 | Not  | **Partially** | 8.8 establish user access and secure credentials management procedures (brief) |
| T5.2.3 | Access Control Policy - CPX policy p15 | Not  | **Partially** | 8.9 no default credentials, encryption/hashing for stored credentials, no cleartext |
| T5.2.4 | Access Control Policy - CPX policy p13 | Not  | **Partially** | 8.7 review access rights at intervals and after promotion/transfer/termination; frequent privileged review |
| T5.4.1 | Access Control Policy - CPX policy p8 | Not  | **Partially** | 8.2 procedure for use of network services (wireless/VPN/third-party), consistent with policy |
| T5.4.7 | Access Control Policy - CPX policy p9 | Not  | **Partially** | 8.3 Wireless Access: authentication protocols, controls, periodic surveys |
| T5.5.1 | Access Control Policy - CPX policy p16 | Not  | **Partially** | 8.10 formal secure logon procedure, banner, strong auth, lockout; no session-timeout atoms |
| T5.6.3 | Access Control Policy - CPX policy p18 | Not  | **Fully Add** | 8.12 publishing guidelines with reviews/approvals, sanitization training, periodic scan for exposed non-public |
| T6.2.2 | Third-Party Security Policy v2.0 U p7 | Not  | **Partially** | 7.1 identify/mandate/monitor third-party security controls and lifecycle |
| T6.2.2 | Third-Party Security Policy v2.0 U p8 | Not  | **Partially** | 7.2.4 service management relationship: monitor service levels, review reports, conduct audits |
| T6.2.3 | Third-Party Security Policy v2.0 U p7 | Part | **Not Addre** | 7.1 general approach; change-management methodology/parameters not addressed (those are in 7.3) |
| T7.7.1 | Vulnerability Management Policy v1 p7 | Not  | **Partially** | 6.1 test/verify vulnerabilities (automated VA/pen tests), remediation plans before production |
| T8.1.1 | Information Security Incident Mana p3 | Full | **Partially** | 1 INTRODUCTION: framework for managing incidents + roles for detection/response/recovery; not full policy - hu |
| T8.1.1 | Information Security Incident Mana p3 | Full | **Partially** | 1 INTRODUCTION: framework + roles for detection/response/recovery; not full policy - human FA over-labels |
| T8.2.2 | Information Security Incident Mana p7 | Not  | **Partially** | 5.1 identify CSIRT personnel, team composition, roles/responsibilities, services; funding/workflow/announce ab |
| T8.2.3 | Information Security Incident Mana p9 | Not  | **Partially** | 5.3 establish incident classification scheme per regulatory issuances; assess/classify events |
| T8.2.3 | Information Security Incident Mana p9 | Not  | **Partially** | 5.3 incident classification scheme per regulatory issuances |
| T8.2.5 | Information Security Incident Mana p11 | Not  | **Partially** | 5.5 Incident Response Testing: procedures, breach simulations, tabletop/functional exercises, sector/national  |
| T8.2.5 | Information Security Incident Mana p11 | Not  | **Partially** | 5.5 Incident Response Testing implements the control |
| T8.2.6 | Information Security Incident Mana p7 | Not  | **Partially** | 5.1 CSIRT provides the incident response support resource (personnel, contact, responsibilities) |
| T8.2.6 | Information Security Incident Mana p7 | Not  | **Partially** | 5.1 CSIRT as incident response support resource |
| T8.2.8 | Information Security Incident Mana p13 | Not  | **Partially** | 5.7 document incidents + incident handling (logging, recording, closing) |
| T8.2.8 | Information Security Incident Mana p13 | Not  | **Partially** | 5.7 incident documentation + handling records |
| T8.3.2 | Information Security Incident Mana p9 | Full | **Not Addre** | 5.3 incident classification; not an event-reporting procedure/channels - human FA over-labels |
| T8.3.3 | Information Security Incident Mana p12 | Not  | **Partially** | 5.6 nominated point of contact (CSIRT), handling of security weakness contributing to incident |
| T8.3.3 | Information Security Incident Mana p12 | Not  | **Partially** | 5.6 nominated point of contact (CSIRT), weakness handling |