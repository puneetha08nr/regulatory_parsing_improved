"""
Generate synthetic FA/PA/NA training examples for all UAE IA controls.
No API calls - generates using templates and control-specific knowledge.
Output: data/07_golden_mapping/synthetic_training_data.json
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data/07_golden_mapping"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Template clauses per control ──────────────────────────────────────────────
# Format: dict[control_id] = {"FA": "...", "PA": "...", "NA": "..."}
# If a control_id is not in CLAUSES, the fallback generator is used.

CLAUSES = {
    # ── M2: Risk Management ──────────────────────────────────────────────────
    "M2.1.1": {
        "FA": "The organisation has an approved Information Security Risk Management Policy that defines the risk management framework, methodology, roles, and responsibilities. The policy requires periodic risk assessments, mandates a risk treatment process, and sets risk acceptance criteria. It is reviewed annually and communicated to all staff.",
        "PA": "A risk management section exists within the Information Security Policy but it is not a standalone policy. The risk management approach is described at a high level without detailed methodology, roles, or risk acceptance criteria.",
        "NA": "No Information Security Risk Management Policy exists. Security risks are addressed reactively without a defined framework or process.",
    },
    "M2.2.1": {
        "FA": "Risk identification is performed using a documented methodology that covers assets, threats, vulnerabilities, and impacts. Risk workshops are facilitated annually for each business unit, supplemented by threat intelligence feeds and vulnerability scan results. A risk register captures all identified risks with owner, date, and source.",
        "PA": "Risk identification occurs as part of project initiation and after incidents, but there is no systematic annual process for all business areas. The risk register exists but has not been fully updated in the past 18 months.",
        "NA": "No formal risk identification process exists. Information security risks are not tracked in a register.",
    },
    "M2.2.2": {
        "FA": "Each identified risk is analysed using a defined 5×5 likelihood/impact matrix with documented rating criteria. Analysis results include threat likelihood, vulnerability severity, asset value, and calculated risk level. Analysis is performed by trained staff and reviewed by the risk owner.",
        "PA": "Risks are rated as High/Medium/Low but the criteria for these ratings are not consistently applied. Analysis relies on individual judgment rather than a standardised matrix, leading to inconsistent results across teams.",
        "NA": "No risk analysis is performed. Identified security issues are treated as either urgent or deferred based on ad hoc management decisions without formal analysis.",
    },
    "M2.2.3": {
        "FA": "Risk evaluation compares analysed risk levels against the organisation's documented risk acceptance criteria. Risks above the acceptance threshold are escalated for treatment. The CISO approves risk acceptance decisions. Evaluation outcomes are documented in the risk register with justification.",
        "PA": "Risks are reviewed informally and prioritised by the security team, but formal risk acceptance criteria are not documented. High risks are escalated to management, but the process for determining what constitutes 'acceptable' risk is inconsistent.",
        "NA": "No risk evaluation process exists. All identified risks are placed in a list but no prioritisation or acceptance decisions are made.",
    },
    "M2.3.1": {
        "FA": "For each risk above the acceptance threshold, a treatment decision is documented: mitigate, transfer (insurance/contract), accept (with justification), or avoid. Treatment decisions are approved by the risk owner and recorded in the risk register. Treatment options are evaluated for cost-effectiveness.",
        "PA": "High risks are addressed with mitigation plans but the range of treatment options (transfer, accept, avoid) is not systematically considered. Risk treatment decisions are not consistently documented with justification.",
        "NA": "Risk treatment is limited to ad hoc responses to security incidents. No systematic evaluation of treatment options for identified risks is conducted.",
    },
    "M2.3.2": {
        "FA": "Controls are selected from the UAE IA control catalogue and mapped to each treated risk in the risk treatment plan. Control selection rationale is documented. New controls are assessed for implementation feasibility before inclusion. The control catalogue is reviewed annually for completeness.",
        "PA": "Controls are selected informally based on team knowledge of common security practices. The link between specific controls and specific risks is not formally documented. Not all selected controls map to the UAE IA control catalogue.",
        "NA": "Controls are implemented based on industry best practice without any formal mapping to identified risks.",
    },
    "M2.3.3": {
        "FA": "A Risk Treatment Plan is maintained for all risks requiring treatment. The plan documents: treatment option, selected controls, responsible owner, implementation timeline, and residual risk estimate. Plan progress is reviewed quarterly by the security team and reported to management.",
        "PA": "A risk treatment plan exists for high-priority risks but medium risks lack documented plans or owners. Plan progress is reviewed annually rather than quarterly.",
        "NA": "No risk treatment plan exists. Risks are identified but treatment is not tracked or managed.",
    },
    "M2.3.4": {
        "FA": "A Statement of Applicability (SoA) is maintained listing all UAE IA controls with their applicability status (applicable/not applicable), justification, and implementation status. The SoA is updated annually and reviewed by the CISO. It serves as the primary reference for the ISMS scope.",
        "PA": "A partial SoA exists covering the most critical controls but does not address the full UAE IA control catalogue. Justifications for exclusions are not documented. The SoA has not been updated in over 18 months.",
        "NA": "No Statement of Applicability exists. The organisation has not mapped implemented controls against the UAE IA control requirements.",
    },
    "M2.3.5": {
        "FA": "Information security objectives are defined annually, aligned with organisational strategy, and measurable with defined KPIs. Objectives cover availability targets, detection time SLAs, training completion rates, and vulnerability remediation timelines. Progress is tracked quarterly and reported to the board.",
        "PA": "Security objectives are referenced in the policy but are not measurable. There are no defined KPIs or regular progress reviews. Objectives have not been updated since initial policy creation.",
        "NA": "No information security objectives are defined. The security programme operates without goals or performance measures.",
    },
    "M2.4.1": {
        "FA": "Risk assessments are reviewed and updated annually and whenever significant changes occur (new systems, incidents, regulatory changes, or organisational restructuring). The review process is documented in the risk management procedure, and all updates are version-controlled in the risk register.",
        "PA": "Risk assessments are reviewed annually for some systems but reviews are not triggered by significant changes. New projects sometimes commence without an updated risk assessment.",
        "NA": "Risk assessments are conducted once (e.g., during initial certification) and not reviewed or updated thereafter.",
    },
    "M2.4.2": {
        "FA": "Risk communication occurs through quarterly risk reports to management, annual all-staff security briefings, and dedicated risk discussions in the Information Security Steering Committee. Risk owners are informed of changes to their risks. Consultation with business units occurs during risk identification workshops.",
        "PA": "Risk reporting is provided to the CISO and security team but broader communication to business unit managers and staff is limited. Risk workshops involve IT staff but rarely include business stakeholders.",
        "NA": "Risk information is not communicated beyond the immediate security team. Business units are unaware of risks relevant to their areas.",
    },

    # ── M3: Awareness and Training ───────────────────────────────────────────
    "M3.1.1": {
        "FA": "An Awareness and Training Policy is approved by management and defines the security awareness and training programme requirements, target audiences, mandatory and role-based training, frequency, and evaluation methods. The policy is reviewed annually.",
        "PA": "Security training is provided but no formal policy governs the training programme. There are no defined requirements for frequency, target audience differentiation, or evaluation of training effectiveness.",
        "NA": "No awareness and training policy or programme exists. Security guidance is communicated informally through manager briefings.",
    },
    "M3.2.1": {
        "FA": "An annual security awareness and training programme is in place covering all employees and contractors. The programme includes: induction training for new joiners, annual refresher training, role-based training for privileged users, and phishing simulation exercises. Completion is tracked and reported to management.",
        "PA": "Annual security awareness training is provided via online modules, but completion rates are low (below 70%) and are not followed up with non-completers. Role-based training for administrators and managers is not in place.",
        "NA": "No security awareness or training programme exists. Security is occasionally mentioned in new employee induction but not covered in ongoing training.",
    },
    "M3.3.1": {
        "FA": "Training needs are assessed annually through a skills gap analysis against defined security competency requirements for each role. Results are used to produce a targeted training plan. Training needs arising from risk assessments, incidents, or regulatory changes are incorporated mid-year.",
        "PA": "Training needs are identified informally by the security team based on perceived gaps and recent incidents. No formal skills gap analysis or competency framework is used.",
        "NA": "Training needs are not assessed. Training content is based on vendor offerings rather than organisational requirements.",
    },
    "M3.3.2": {
        "FA": "An annual training implementation plan is produced, specifying training courses, target audiences, delivery methods, timelines, and budget. The plan aligns with the training needs assessment results. Progress against the plan is reviewed quarterly.",
        "PA": "A training calendar exists listing scheduled sessions but it is not tied to a formal training needs assessment. The calendar is updated reactively rather than based on a systematic plan.",
        "NA": "No training implementation plan exists. Training is arranged ad hoc when budget is available.",
    },
    "M3.3.3": {
        "FA": "Training is delivered through multiple channels (e-learning, instructor-led, and on-the-job). Delivery is tracked by the HR and security teams. All mandatory courses require a minimum pass mark of 80%. Failed attempts are followed up with remedial training.",
        "PA": "Training is delivered via e-learning modules. Completion is tracked but pass mark requirements are not enforced. Instructors are not available for role-based technical training.",
        "NA": "Training delivery is inconsistent. Some departments receive security briefings and others receive none.",
    },
    "M3.3.4": {
        "FA": "Training effectiveness is measured through pre/post assessments, phishing click-rate trends, and periodic security knowledge surveys. Results are analysed annually to identify areas requiring improved content or additional training. Findings are reported to management.",
        "PA": "Post-training satisfaction surveys are collected but are not analysed for learning effectiveness. There is no link between training outcomes and security incident trends.",
        "NA": "Training results are not measured. Once training is delivered it is considered complete regardless of comprehension.",
    },
    "M3.3.5": {
        "FA": "Training attendance records, completion certificates, and assessment scores are maintained in the HR learning management system for all staff. Records are retained for a minimum of three years. Audit evidence of training completion is available on request.",
        "PA": "Training completion is tracked in a spreadsheet, but it is not integrated with HR systems. Records for contractors and temporary staff are incomplete.",
        "NA": "No training records are maintained. There is no evidence of who has or has not received security training.",
    },
    "M3.4.1": {
        "FA": "An annual security awareness campaign is conducted using multiple channels: posters, intranet articles, email newsletters, and security awareness events. Campaign themes are aligned with current threat intelligence and organisational risk priorities. Campaign effectiveness is measured through staff surveys.",
        "PA": "Ad hoc awareness communications are sent in response to incidents or new threats (e.g., phishing alerts), but there is no structured annual awareness campaign with defined themes, targets, or measurement.",
        "NA": "No security awareness campaign is conducted. Staff receive no regular security communications beyond onboarding.",
    },

    # ── M4: Human Resources Security ─────────────────────────────────────────
    "M4.1.1": {
        "FA": "A Human Resources Security Policy covering pre-employment, during employment, and termination phases is approved and communicated to all HR and management staff. The policy defines screening requirements, employment terms for security, and obligations upon termination. It is reviewed annually.",
        "PA": "Security is addressed in the general HR policy but a dedicated Human Resources Security Policy does not exist. Pre-employment screening requirements are documented but termination security obligations are incomplete.",
        "NA": "No HR security policy exists. Security considerations are not formally integrated into HR processes.",
    },
    "M4.2.1": {
        "FA": "All employees and contractors with access to sensitive systems or data undergo background screening proportionate to their role. Screening includes identity verification, employment history, criminal record check (where legally permitted), and reference checks. Screening is completed before access is granted.",
        "PA": "Background checks are performed for senior roles and those with privileged access, but they are not consistently required for all staff or contractors with access to sensitive data. The screening process is not formally documented.",
        "NA": "No background screening is conducted prior to employment or contractor engagement.",
    },
    "M4.2.2": {
        "FA": "Employment contracts and contractor agreements include explicit information security obligations: confidentiality requirements, acceptable use provisions, data handling obligations, and consequences for policy violations. Agreements are signed before access to any system is granted.",
        "PA": "Employment contracts include a general confidentiality clause, but detailed security obligations (acceptable use, data handling) are not specified in the contract. Contractor agreements do not consistently include security terms.",
        "NA": "Employment contracts do not include information security obligations. Security responsibilities are communicated verbally during induction.",
    },
    "M4.3.1": {
        "FA": "Line managers are accountable for ensuring their teams comply with information security policies. Management responsibilities are documented in the security policy and referenced in manager job descriptions. Managers receive specific security management training and are included in security incident escalation paths.",
        "PA": "Managers are expected to enforce security policies but these responsibilities are not formally documented in job descriptions or management guidelines. Security training for managers is the same general course provided to all staff.",
        "NA": "No management security responsibilities are defined. Security enforcement is left entirely to the IT/security team.",
    },
    "M4.3.2": {
        "FA": "A disciplinary process for security policy violations is documented and consistently applied. The process includes: investigation, evidence gathering, proportionate sanctions (up to dismissal for serious violations), and appeal rights. HR and legal are involved for serious violations. The process is communicated to all staff.",
        "PA": "The general HR disciplinary process applies to security violations but there is no specific guidance on security-related disciplinary matters. Sanctions for minor violations are not defined, leading to inconsistent application.",
        "NA": "No disciplinary process for security violations exists. Policy breaches are addressed informally or ignored.",
    },
    "M4.4.1": {
        "FA": "Termination and change of role procedures include security steps: removal of access within 24 hours, collection of assets, completion of security exit checklist, and revocation of privileged access immediately upon notification. HR notifies IT and security of all terminations before the employee's last day.",
        "PA": "Offboarding checklists exist but are not always completed fully. Access revocation sometimes occurs days after the last day of employment, particularly for contractor terminations.",
        "NA": "No formal termination security procedures exist. Account deactivation depends on IT being notified informally, and the timing is inconsistent.",
    },
    "M4.4.2": {
        "FA": "All organisation-issued assets (laptops, phones, tokens, access cards) are itemised in the asset register and collected upon termination. The collection is documented in the exit checklist signed by both the employee and HR. A final check verifies all items are returned before final payment is processed.",
        "PA": "A basic exit checklist includes return of laptop and access card, but other assets (tokens, mobile devices, corporate credit cards with access) are not consistently inventoried or collected.",
        "NA": "No process exists for collecting assets from leavers. Equipment is returned or not returned based on informal reminders.",
    },
    "M4.4.3": {
        "FA": "All access rights are reviewed and revoked upon termination or role change. Privileged access is revoked immediately; standard access within 24 hours. Access revocation is tracked through the identity management system and confirmed by the security team. Access reinstatement for returning employees requires re-authorisation.",
        "PA": "Active directory accounts are disabled upon termination, but shared credentials, application-specific accounts, and physical access rights are not consistently updated. Role-change access adjustments rely on manual requests from managers.",
        "NA": "Access rights are not systematically revoked. Former employee accounts may remain active for extended periods after departure.",
    },

    # ── M5: Compliance ────────────────────────────────────────────────────────
    "M5.1.1": {
        "FA": "A Compliance Policy defines requirements for identifying and meeting legal, regulatory, contractual, and internal information security obligations. The policy assigns compliance monitoring responsibilities, mandates a compliance register, and requires annual compliance reviews. The policy is approved by legal counsel and the CISO.",
        "PA": "Compliance requirements are addressed within the general Information Security Policy but a dedicated Compliance Policy does not exist. Responsibilities for monitoring compliance with specific regulations are not clearly assigned.",
        "NA": "No Compliance Policy exists. Legal and regulatory security requirements are managed informally without a systematic approach.",
    },
    "M5.2.1": {
        "FA": "A legal and regulatory requirements register is maintained, covering all applicable information security legislation, regulations, and contractual obligations (e.g., UAE Cybersecurity Law, data protection regulations, sector regulations). The register is reviewed quarterly and updated when regulatory changes occur. A compliance owner is assigned for each requirement.",
        "PA": "Key legal requirements (data protection, cybersecurity law) are known to the legal and compliance team but are not documented in a formal register. Some regulatory requirements applicable to specific systems or processes are not tracked.",
        "NA": "No register of applicable laws and regulations exists. Legal and regulatory compliance is managed reactively when issues arise.",
    },
    "M5.2.2": {
        "FA": "An Intellectual Property Rights (IPR) policy governs the use of software, data, and other intellectual property. The policy mandates software licence management, prohibits use of unlicensed software, and establishes a software asset register. Compliance is verified through annual software audits.",
        "PA": "Software licensing is managed centrally for standard applications, but shadow IT and department-purchased software are not consistently tracked. There is no formal IPR policy addressing all categories of intellectual property.",
        "NA": "No IPR controls exist. Software is installed and used without licence verification. No software asset register is maintained.",
    },
    "M5.2.3": {
        "FA": "An records management policy defines retention periods, storage requirements, and disposal procedures for organisational records, including security-relevant records (logs, policies, contracts). Records are classified and stored in approved systems. Retention periods align with legal requirements. Disposal is documented.",
        "PA": "Retention policies exist for financial and HR records but information security records (audit logs, incident reports, risk assessments) do not have defined retention periods. Disposal of records is not systematically documented.",
        "NA": "No records management policy or retention schedule exists for information security records. Records are kept or deleted based on storage capacity.",
    },
    "M5.2.4": {
        "FA": "Personal data is processed in accordance with applicable data protection regulations. A privacy policy, data inventory, and privacy impact assessment process are in place. A designated Data Protection Officer (or equivalent) oversees compliance. Data subject rights are addressed and response procedures are documented.",
        "PA": "A privacy policy is in place and data subjects can submit requests, but the data inventory is incomplete and privacy impact assessments are not consistently conducted for new processing activities.",
        "NA": "No data protection or privacy programme exists. Personal data is processed without a documented legal basis or subject rights management.",
    },
    "M5.2.5": {
        "FA": "An Acceptable Use Policy (AUP) clearly defines permitted and prohibited uses of information systems. The AUP covers internet use, email, removable media, and personal device usage. Monitoring of system use is disclosed to users. Policy violations are subject to the disciplinary process. All staff sign the AUP annually.",
        "PA": "An AUP exists but it has not been updated in three years and does not cover cloud services, mobile devices, or remote working scenarios that are now prevalent. Monitoring activities are not disclosed in the AUP.",
        "NA": "No Acceptable Use Policy exists. System use is unrestricted and unmonitored.",
    },
    "M5.2.6": {
        "FA": "A Cryptographic Controls Policy mandates the use of approved encryption algorithms (e.g., AES-256, RSA-2048), key lengths, and protocols for data at rest and in transit. Use of deprecated algorithms (e.g., DES, MD5, RC4) is prohibited. The policy addresses export control compliance for cryptographic tools.",
        "PA": "Encryption is used for sensitive data but there is no formal policy defining approved algorithms or key management requirements. Some legacy systems use deprecated cryptographic protocols that have not been updated.",
        "NA": "No policy on cryptographic controls exists. Encryption decisions are made ad hoc by individual developers and system administrators.",
    },
    "M5.2.7": {
        "FA": "The organisation's obligations to the information sharing community are documented, including confidentiality of received intelligence, traffic light protocol (TLP) compliance, and prohibition on sharing received data beyond authorised recipients. All staff involved in information sharing receive specific training on these obligations.",
        "PA": "Staff involved in information sharing are aware of TLP markings in general terms, but obligations are not formally documented and training on community-specific rules is not provided.",
        "NA": "No documented obligations to information sharing communities exist. Shared threat intelligence is used and forwarded without adherence to community rules.",
    },
    "M5.3.1": {
        "FA": "Compliance with information security policies and standards is verified through a programme of management reviews, internal audits, and self-assessments. Policy owners are accountable for compliance in their areas. Non-compliance findings are tracked in a register and escalated for remediation with defined timelines.",
        "PA": "Policy compliance is checked during internal audits but the audit cycle does not cover all policy areas annually. Non-compliance findings are raised but remediation is not consistently tracked to closure.",
        "NA": "No mechanism for checking compliance with information security policies exists. Policy adherence relies entirely on voluntary compliance.",
    },
    "M5.4.1": {
        "FA": "Technical compliance checks are conducted using vulnerability scanners, configuration compliance tools (e.g., CIS benchmarks), and periodic penetration tests. Results are reported to the CISO and tracked for remediation. Critical findings are remediated within defined SLAs (critical: 72h, high: 30 days).",
        "PA": "Vulnerability scans are conducted quarterly for servers and network devices, but endpoints and applications are not consistently scanned. Penetration testing is conducted on an ad hoc basis when budget allows. Remediation SLAs are defined but not consistently enforced.",
        "NA": "No technical compliance checking is performed. System configurations and vulnerabilities are not assessed against security standards.",
    },
    "M5.5.1": {
        "FA": "Information systems audits are planned, resourced, and conducted independently of the operational teams being audited. The audit scope, timing, and access requirements are agreed in advance. Audit findings are reported to management and tracked for remediation.",
        "PA": "IT audits are conducted by the internal audit team but audit independence is limited for security-specific audits where IT staff are involved in both operations and audit activities.",
        "NA": "No information systems audits are conducted.",
    },
    "M5.5.2": {
        "FA": "Audit tools (vulnerability scanners, forensic tools, compliance check scripts) are stored in a restricted repository accessible only to authorised auditors. Tools are version-controlled and protected from unauthorised modification. Access to audit tools is logged and reviewed.",
        "PA": "Audit tools are stored on a dedicated audit workstation, but access controls are shared among the security team rather than restricted to auditors. Tools are not version-controlled.",
        "NA": "Audit tools are installed on standard workstations accessible to IT staff. No access controls or version management are applied.",
    },
    "M5.5.3": {
        "FA": "The organisation conducts audits of community functions (shared platforms, information sharing hubs) in which it participates to verify that community-level controls meet its security requirements. Audit findings related to community platforms are escalated to community administrators.",
        "PA": "Community platforms are assessed during initial onboarding but ongoing audits of community functions are not conducted. Changes to community platform security are not monitored.",
        "NA": "No audits of community functions are conducted. The organisation assumes community platforms are adequately secured without verification.",
    },

    # ── M6: Performance Evaluation ────────────────────────────────────────────
    "M6.1.1": {
        "FA": "A Performance Evaluation Policy defines the approach for monitoring, measuring, analysing, and evaluating ISMS performance. The policy specifies what is to be monitored, measurement methods, frequency of evaluation, and who is responsible for analysis and reporting. It is reviewed annually.",
        "PA": "Security performance is monitored informally, but no formal policy defines what metrics are tracked, how they are measured, or how frequently performance is reviewed.",
        "NA": "No performance evaluation policy or practice exists for information security.",
    },
    "M6.2.1": {
        "FA": "An ISMS performance dashboard tracks KPIs including: vulnerability remediation rates, training completion, incident response times, audit findings closed, and patch deployment rates. Metrics are reviewed monthly by the security team and quarterly by management. Trends are analysed to identify improvement opportunities.",
        "PA": "Some security metrics are collected (e.g., number of incidents, training completion) but they are not consolidated into a dashboard or reviewed systematically. Management reporting on security performance is infrequent.",
        "NA": "No security metrics are collected or analysed. ISMS performance is not measured.",
    },
    "M6.2.2": {
        "FA": "Internal ISMS audits are conducted annually by qualified auditors who are independent of the areas being audited. The audit programme covers all ISMS scope areas over a three-year cycle. Audit criteria, scope, and methods are documented. Findings are reported to top management.",
        "PA": "Internal audits are conducted but focus on IT controls rather than the broader ISMS. Not all ISMS scope areas are covered within any regular cycle. Auditor independence is not always ensured.",
        "NA": "No internal ISMS audit programme exists.",
    },
    "M6.3.1": {
        "FA": "A corrective action process ensures that nonconformities are investigated for root cause, corrective actions are defined and implemented, and effectiveness is verified before closure. Corrective actions are tracked in a register with owners and target dates. Recurring nonconformities trigger systemic review.",
        "PA": "Corrective actions are raised for audit findings and incidents, but root cause analysis is not consistently performed. Some actions are closed based on planned implementation rather than verified effectiveness.",
        "NA": "No formal corrective action process exists. Issues are addressed when they cause visible problems but root causes are not investigated.",
    },
    "M6.3.2": {
        "FA": "Continual improvement of the ISMS is driven by: lessons learned from incidents, audit results, management reviews, performance metrics, and changes in the threat landscape. Improvement actions are tracked in the ISMS improvement register. The CISO presents improvement plans to management annually.",
        "PA": "Improvements to security controls are made reactively following incidents or audit findings. There is no proactive improvement programme driven by performance trends or threat intelligence.",
        "NA": "The ISMS is static. Controls are not updated unless a serious incident forces a change.",
    },

    # ── T1: Asset Management ──────────────────────────────────────────────────
    "T1.1.1": {
        "FA": "An Asset Management Policy covers all information assets including hardware, software, data, and services. The policy defines asset classification, ownership, acceptable use, and disposal requirements. It is approved by management and reviewed annually.",
        "PA": "Asset management guidance is included in the general IT policy but a standalone Asset Management Policy does not exist. The policy does not address information assets owned by business units or cloud-hosted assets.",
        "NA": "No Asset Management Policy exists.",
    },
    "T1.2.1": {
        "FA": "A comprehensive asset inventory is maintained covering all hardware, software, data assets, and cloud services. Each asset has a designated owner, classification level, and location. The inventory is updated whenever assets are acquired or decommissioned. Automated discovery tools supplement manual processes.",
        "PA": "An IT asset register exists for hardware and licensed software but cloud services, shadow IT, and information assets (databases, documents) are not inventoried. The inventory is updated manually and may be out of date.",
        "NA": "No asset inventory is maintained. The organisation cannot enumerate its information assets.",
    },
    "T1.2.2": {
        "FA": "All information assets have a designated owner responsible for the asset's protection, classification review, and access authorisation. Ownership is recorded in the asset register. Owners are notified of their responsibilities during asset registration and via the asset management policy.",
        "PA": "Asset ownership is assigned for major systems but not for all assets in the register. Business-unit-owned information assets (documents, databases, SaaS tools) often lack a designated owner.",
        "NA": "No formal asset ownership programme exists. Responsibility for asset protection is not assigned to individuals.",
    },
    "T1.2.3": {
        "FA": "Acceptable Use of Assets Policy defines permitted uses for all asset categories (devices, removable media, cloud storage, personal devices). The policy is communicated during induction and acknowledged annually. Technical controls enforce key provisions (e.g., web filtering, USB restrictions).",
        "PA": "An Acceptable Use Policy exists but was last updated two years ago and does not cover cloud services or personal device use (BYOD). Technical controls enforce some provisions (web filtering) but USB ports are unrestricted.",
        "NA": "No Acceptable Use Policy exists. Employees use organisational assets for any purpose without restriction.",
    },
    "T1.2.4": {
        "FA": "A BYOD policy governs all personal devices used for business purposes. The policy requires device enrolment in MDM, minimum security standards (PIN, encryption, remote wipe capability), and separation of personal and corporate data through containerisation. BYOD users sign an acceptance agreement.",
        "PA": "Employees use personal devices for business email and messaging but there is no formal BYOD policy or MDM enrolment. Some guidance on password-protecting personal devices has been issued but is not enforced.",
        "NA": "Personal devices are used for business purposes without any policy, controls, or enrolment requirement.",
    },
    "T1.3.1": {
        "FA": "An information classification scheme with four levels (Public, Internal, Confidential, Secret) is defined in the Information Classification Policy. Classification criteria and examples are documented. All information assets are classified by their owner. Classification assignments are reviewed when assets are substantially changed.",
        "PA": "A three-level classification scheme exists (Public, Internal, Restricted) but classification of individual assets is not consistently applied. Many documents and datasets are unclassified, and the criteria for distinguishing Restricted from Internal are unclear to users.",
        "NA": "No information classification scheme exists. All information is handled uniformly regardless of sensitivity.",
    },
    "T1.3.2": {
        "FA": "Information labelling requirements are defined per classification level. Physical documents include classification headers and footers; electronic documents include classification metadata and visible markings. Labelling is enforced through document templates and data loss prevention (DLP) tools that prompt users to classify before sending.",
        "PA": "Classification labels are applied to documents when staff remember to do so, but there is no enforcement mechanism. DLP rules are not configured to require labelling before transmission. Physical document labelling is inconsistent.",
        "NA": "No information labelling is in place. Documents do not carry classification markings.",
    },
    "T1.3.3": {
        "FA": "Handling procedures for each classification level are documented: transmission methods (encrypted for Confidential/Secret), storage requirements (encrypted at rest for Restricted+), retention and disposal procedures. Staff receive training on handling requirements for each level.",
        "PA": "Handling guidelines exist for the highest classification level but are not documented for lower levels. Training on handling requirements is included in general security awareness but is not classification-specific.",
        "NA": "No information handling procedures exist. Information is handled without regard for its classification or sensitivity.",
    },
    "T1.4.1": {
        "FA": "A Removable Media Management Policy prohibits use of unapproved removable media. Approved media is encrypted and inventoried. USB port control software enforces policy on all endpoints. Media used to transfer sensitive data is degaussed or physically destroyed before disposal.",
        "PA": "USB use is restricted by policy but technical enforcement is not in place on all endpoints. Staff are aware of the policy but non-compliant use is observed and not consistently addressed.",
        "NA": "Removable media is used freely without policy, inventory, or encryption requirements.",
    },
    "T1.4.2": {
        "FA": "Media disposal follows a documented procedure based on data sensitivity: overwriting for lower classifications, cryptographic erasure or physical destruction for higher classifications. A certificate of destruction is obtained for outsourced disposal. Disposal records are maintained for audit purposes.",
        "PA": "End-of-life equipment is sent to a certified refurbisher, but there is no formal verification of data erasure before handover. Internal media disposal (CD/DVD, paper) does not follow a structured process.",
        "NA": "Media is disposed of without secure erasure. Old hard drives are donated or discarded with data intact.",
    },

    # ── T2: Physical and Environmental Security ───────────────────────────────
    "T2.1.1": {
        "FA": "A Physical and Environmental Security Policy is approved by management and covers all facilities. The policy defines physical security zones, access control requirements, visitor management, environmental controls, and clear desk requirements. It is reviewed annually.",
        "PA": "Physical security requirements are addressed in operational procedures for the data centre, but no overarching policy covers all facilities including offices and remote sites.",
        "NA": "No Physical and Environmental Security Policy exists.",
    },
    "T2.2.1": {
        "FA": "Physical security perimeters are defined for all facilities processing or storing sensitive information. Perimeters are secured with locked doors, security barriers, and CCTV. Entry points are minimised and controlled. Security perimeter boundaries are reviewed when facilities change.",
        "PA": "The data centre has a clearly defined and secured perimeter, but office areas housing workstations and paper records have less rigorous physical security with shared lobby access.",
        "NA": "No physical security perimeters are defined. Facilities are accessible without security checks.",
    },
    "T2.2.2": {
        "FA": "Physical entry controls include swipe card access with individual credentials, visitor log, escorted visitor policy, and CCTV at all entry points. Access rights are reviewed quarterly and revoked within 24 hours of termination. Tailgating is addressed through mantraps at high-security areas.",
        "PA": "Card access is in place at main entry points, but not all secure areas require individual credentials. Visitor logs are maintained but visitors are not always escorted. CCTV coverage has blind spots.",
        "NA": "Physical entry to facilities is not controlled. Areas processing sensitive information are accessible to all building occupants.",
    },
    "T2.2.3": {
        "FA": "Server rooms and offices processing confidential information are secured with locked doors and role-appropriate access controls. Security requirements for each area are documented. Working in secure areas requires prior authorisation and is logged. Unsupervised access by maintenance contractors is prohibited.",
        "PA": "Server rooms are locked but multiple staff have access for operational reasons beyond those with a documented need. Offices with confidential records are not separately secured from general work areas.",
        "NA": "Offices and server rooms are not specifically secured. All staff can access all areas without restriction.",
    },
    "T2.2.4": {
        "FA": "Facilities are assessed for external and environmental threats (flooding, fire, power, physical attack) during the physical security review. Mitigations include fire suppression systems, flood barriers, UPS, and blast-resistant construction for critical facilities. Risks are reviewed annually.",
        "PA": "Fire suppression is in place in the data centre but no formal assessment of flooding, physical attack, or extreme weather risks has been conducted. Some facilities are in high-flood-risk locations without flood mitigation.",
        "NA": "Environmental and external threats to facilities have not been assessed or mitigated.",
    },
    "T2.2.5": {
        "FA": "Working in secure areas is governed by documented procedures: prohibition of personal cameras and devices, need-to-know access, no unsupervised access for visitors, and clean desk enforcement. Procedures are communicated to all staff with access. Compliance is checked during physical security reviews.",
        "PA": "Some secure area procedures exist for the data centre, but general office procedures (e.g., clean desk, personal device restrictions) are guidelines rather than enforced rules.",
        "NA": "No procedures for working in secure areas are defined.",
    },
    "T2.2.6": {
        "FA": "Public access, delivery, and loading areas are physically separated from secure processing areas. Deliveries are inspected before being brought into secure areas. Access from loading areas to the main facility requires escort. CCTV covers all loading/delivery areas.",
        "PA": "A delivery reception area exists but is adjacent to the server room with no formal separation or inspection procedure for incoming packages.",
        "NA": "No controls exist for public access areas, deliveries, or loading zones.",
    },
    "T2.3.1": {
        "FA": "Critical equipment is sited to minimise physical risks: raised floors to prevent flooding, adequate separation for airflow, UPS protection, and restricted access. Environmental monitoring (temperature, humidity, water) is deployed with automated alerting. Equipment placement is documented and reviewed annually.",
        "PA": "Server racks have UPS protection and air conditioning, but environmental monitoring only covers temperature. Equipment placement was not formally assessed for physical risks (flooding, vibration from adjacent equipment).",
        "NA": "Equipment is placed without consideration of physical risks. No environmental monitoring is deployed.",
    },
    "T2.3.2": {
        "FA": "Supporting utilities (power, cooling, water, fire suppression) are documented, maintained, and tested. Redundancy is provided for critical utilities. UPS provides short-term backup and a diesel generator provides extended backup. Utility failures trigger alerts and a documented response procedure.",
        "PA": "UPS is in place for the data centre, but there is no generator backup for extended outages. Air conditioning is single-unit with no redundancy. Utility maintenance is conducted reactively.",
        "NA": "No assessment or documentation of supporting utility requirements. Power and cooling failures would cause immediate outage.",
    },
    "T2.3.3": {
        "FA": "Power and data cables are protected from interference and damage: power cables are routed separately from data cables, cables are labelled, and cable runs are documented. Cables in public areas are in conduit. Inspections of cable runs are conducted annually.",
        "PA": "Data and power cables are generally separated but cabling in some areas is disorganised and unlabelled. Cables in public areas are not consistently protected by conduit.",
        "NA": "No cabling security controls are in place. Cables are not documented or protected.",
    },
    "T2.3.4": {
        "FA": "Equipment maintenance is conducted according to manufacturer recommendations and recorded in the asset management system. Maintenance is performed by authorised personnel only. Off-site maintenance requires data sanitisation of equipment before dispatch. Maintenance records include date, technician, and work performed.",
        "PA": "Server hardware is maintained under vendor support contracts, but maintenance of networking equipment and other hardware is reactive and not formally recorded. Data sanitisation before off-site maintenance is not consistently performed.",
        "NA": "No equipment maintenance programme exists. Equipment is used until failure with no preventive maintenance.",
    },
    "T2.3.5": {
        "FA": "Equipment taken off-site is covered by an off-premises use policy requiring manager approval, encryption of data, and insurance coverage. A register of off-premises equipment is maintained. Return dates are tracked and equipment is inspected upon return.",
        "PA": "Laptops taken off-site are encrypted, but there is no formal approval process or off-premises equipment register. Employees take other equipment (portable drives, test hardware) off-site without logging.",
        "NA": "Equipment is taken off-site without controls, approval, or documentation.",
    },
    "T2.3.6": {
        "FA": "Equipment disposal and reuse follows documented procedures: data overwriting using approved tools (meeting NIST 800-88 standards) before reuse, or physical destruction before disposal. Certificates of destruction are obtained. Disposal is restricted to approved vendors.",
        "PA": "Laptops are factory reset before reuse or disposal, but lower-level formatting rather than certified overwriting is used. Network equipment and storage media are not consistently sanitised before disposal.",
        "NA": "Equipment is disposed of or reused without data erasure.",
    },
    "T2.3.7": {
        "FA": "Removal of equipment, media, or information from the facility requires written authorisation from an appropriate manager and is logged. Spot checks of outgoing items are conducted. The removal authorisation process is communicated to all staff and enforced at building exits.",
        "PA": "High-value equipment removal is controlled but removal of smaller items (USB drives, laptops) is not consistently logged. Spot checks of outgoing items are not conducted.",
        "NA": "No controls exist for removal of equipment or information from the organisation's premises.",
    },
    "T2.3.8": {
        "FA": "Unattended workstation policy requires screen lock after 5 minutes of inactivity and full logout at end of day. Group Policy enforces these settings on all Windows devices. Users are reminded of the policy through awareness campaigns. Non-compliance is addressed under the disciplinary procedure.",
        "PA": "Screen lock is configured on most workstations but the timeout period exceeds the policy requirement (15 minutes instead of 5). Some shared workstations do not require individual login.",
        "NA": "No controls for unattended user equipment are in place. Workstations are left unlocked and logged in when unattended.",
    },
    "T2.3.9": {
        "FA": "A Clear Desk and Clear Screen Policy mandates removal of sensitive materials from desks when not in use, locked cabinets for sensitive documents, and workstation screen locks. Physical inspections are conducted quarterly. Non-compliance is reported to management.",
        "PA": "A clean desk policy exists but is not consistently enforced. Physical inspections are not conducted, and sensitive documents are regularly observed left on desks outside business hours.",
        "NA": "No clear desk or clear screen policy exists.",
    },

    # ── T3: Operations Management ─────────────────────────────────────────────
    "T3.1.1": {
        "FA": "An Operations Management Policy is approved and covers: documented procedures, change management, segregation of duties, capacity management, malware controls, backup, and monitoring. The policy is communicated to all IT operations staff and reviewed annually.",
        "PA": "Operational security requirements are addressed in individual IT procedures (change management, backup), but an overarching Operations Management Policy does not exist to coordinate these activities.",
        "NA": "No Operations Management Policy exists.",
    },
    "T3.2.1": {
        "FA": "Common Systems Configuration Guidelines (hardening standards) are published for all supported OS and application platforms, based on CIS Benchmarks. Compliance with the standards is verified at deployment and monitored via configuration compliance tools. Deviations require documented exception approval.",
        "PA": "Hardening guidelines exist for servers but not for workstations, network devices, or cloud workloads. Guidelines are applied manually at deployment but compliance is not continuously monitored.",
        "NA": "No system configuration guidelines exist. Systems are deployed with default or developer-chosen configurations.",
    },
    "T3.2.2": {
        "FA": "Documented operating procedures cover all key operational tasks (startup/shutdown, backup procedures, error handling, maintenance tasks). Procedures are version-controlled, reviewed annually, and accessible to on-call staff. New procedures are required for all changes to critical system operations.",
        "PA": "Procedures exist for major operational tasks (backup, patch deployment) but many routine tasks are undocumented. On-call staff rely on tribal knowledge rather than written procedures.",
        "NA": "No documented operating procedures exist. Operational tasks are performed based on individual knowledge.",
    },
    "T3.2.3": {
        "FA": "A formal change management process requires: change request, impact and risk assessment, security review for significant changes, approval, testing in non-production, and post-implementation review. Emergency change procedures are defined separately. All changes are logged in the change management system.",
        "PA": "Change management is in place for infrastructure changes but is not consistently applied to application changes or cloud configuration changes. Security impact assessments are not always completed for changes.",
        "NA": "No formal change management process exists. System changes are made directly to production without testing or approval.",
    },
    "T3.2.4": {
        "FA": "Segregation of duties is enforced for critical functions: development, test, and production access are separate; financial transaction approval and posting are split; privileged access provisioning and approval are distinct roles. Compensating controls (enhanced logging) are applied where full segregation is not practical.",
        "PA": "Segregation of duties is applied in the financial system but not consistently in IT operations. Developers have access to production systems for debugging purposes without a compensating control.",
        "NA": "No segregation of duties controls exist. Individual employees can initiate, approve, and complete critical security functions without oversight.",
    },
    "T3.2.5": {
        "FA": "Development, test, and production environments are physically and logically separated. Production data is not used in development or test without anonymisation. Access to production is restricted to authorised operations staff. Promotion of changes between environments follows the change management process.",
        "PA": "Development and production are on separate servers, but the test environment shares infrastructure with production. Production data is sometimes copied to test for debugging without sanitisation.",
        "NA": "Development, test, and production environments are co-located. Developers have unrestricted access to production systems.",
    },
    "T3.3.1": {
        "FA": "Capacity requirements for critical systems are monitored against defined thresholds. Capacity planning is conducted annually with three-year projections. Alerts trigger when utilisation exceeds 80% of capacity. Capacity constraints are reported to management and addressed in the infrastructure investment plan.",
        "PA": "System performance is monitored and reactive scaling is applied when bottlenecks are identified, but no formal capacity planning or proactive threshold management is in place.",
        "NA": "No capacity monitoring or planning is conducted. Systems are expanded only after performance degradation occurs.",
    },
    "T3.3.2": {
        "FA": "Formal system acceptance testing is conducted before production deployment. Testing criteria are defined, including security testing (vulnerability scan, OWASP top 10 for applications). A signed acceptance document is required from the business owner before deployment. Test results are retained.",
        "PA": "Functional testing is conducted before deployment but security testing is not a consistent part of the acceptance process. Acceptance is based on functional sign-off without a formal security review.",
        "NA": "No formal system acceptance testing process exists. Systems are deployed when development considers them ready.",
    },
    "T3.4.1": {
        "FA": "Anti-malware controls are deployed on all endpoints, servers, and email gateways. Definitions are updated daily. Real-time scanning is enabled. Detected malware is automatically quarantined and reported. Anti-malware coverage is monitored centrally, and non-compliant endpoints are isolated.",
        "PA": "Anti-malware is deployed on workstations and servers but mobile devices are not covered. Definition updates occur weekly rather than daily. Detected malware reports are not centrally monitored.",
        "NA": "No anti-malware controls are deployed organisation-wide.",
    },
    "T3.5.1": {
        "FA": "A backup policy defines backup frequency (daily incremental, weekly full), retention periods (30 days on-site, 12 months off-site), and recovery objectives (RTO 4 hours, RPO 24 hours). Backups are encrypted and stored in a geographically separate location. Restoration tests are conducted quarterly.",
        "PA": "Daily backups are performed for servers, but backup coverage of workstations and SaaS data is incomplete. Backups are stored on-site only. Restoration testing is conducted annually rather than quarterly.",
        "NA": "No formal backup policy or regular backup process exists. Data recovery from system failure is not possible.",
    },
    "T3.6.1": {
        "FA": "A Monitoring Policy and Procedures document defines what is monitored (authentication, network traffic, system events, user activity), monitoring frequency, alert thresholds, and responsibilities for review. The policy requires retention of monitoring data for 12 months and is reviewed annually.",
        "PA": "Monitoring activities are in place for servers and perimeter devices but the scope, alert thresholds, and review responsibilities are not formally documented in a monitoring policy.",
        "NA": "No monitoring policy or procedures exist. Monitoring is ad hoc and inconsistent.",
    },
    "T3.6.2": {
        "FA": "Audit logging is enabled for all critical systems: authentication events, privilege escalation, system configuration changes, and data access events. Logs are centralised in a SIEM. Log coverage is reviewed quarterly and gaps are remediated. Logging policies are aligned with the organisation's data classification requirements.",
        "PA": "Audit logging is enabled on servers and key applications but coverage is incomplete for network devices and cloud workloads. Log events are not consistently forwarded to a centralised system, making correlation difficult.",
        "NA": "Audit logging is not systematically enabled. Security-relevant events are not captured for analysis.",
    },
    "T3.6.3": {
        "FA": "System use is monitored against the Acceptable Use Policy. SIEM rules detect anomalous usage patterns (after-hours access, large data transfers, repeated failed authentications). Alerts are investigated by the security team within defined SLAs. Monitoring results are reported monthly.",
        "PA": "System event logs are reviewed periodically but automated alerting for anomalous behaviour is limited. After-hours access and large data transfers are not automatically flagged for review.",
        "NA": "System use is not monitored. User activity on organisational systems is not analysed.",
    },
    "T3.6.4": {
        "FA": "Log integrity is protected by: forwarding logs to a centralised log server that operations staff cannot modify, hash-based integrity verification, and role-based access control to log management systems. Log tampering attempts generate immediate alerts. Log retention and integrity controls are tested annually.",
        "PA": "Logs are forwarded to a central server but system administrators who manage both servers and the log system can modify or delete logs. Hash-based integrity verification is not implemented.",
        "NA": "Logs are stored locally on the systems they originate from, where administrators have full access to modify or delete them.",
    },
    "T3.6.5": {
        "FA": "Administrator and operator activities are logged separately from user activity logs and stored in a tamper-resistant system. Privileged user logs are reviewed weekly by the security team. Anomalous administrator activity triggers an immediate security investigation.",
        "PA": "Administrator actions are logged within system audit logs, but administrator logs are not separated from general system logs or reviewed with higher scrutiny.",
        "NA": "Administrator activities are not logged or are logged in the same system as user activities without additional scrutiny.",
    },
    "T3.6.6": {
        "FA": "System faults and errors are logged and reviewed daily. A fault management procedure defines fault severity levels, escalation paths, and resolution timelines. Recurring faults are investigated for root causes. Fault trends are reported to management monthly.",
        "PA": "Fault logging is enabled but logs are reviewed reactively when problems are reported rather than proactively. Fault trends are not analysed.",
        "NA": "No fault logging is in place. System errors are only investigated when they cause a visible outage.",
    },
    "T3.6.7": {
        "FA": "All systems synchronise time to a centralised NTP server which is itself synchronised to a reliable time source (e.g., national time server or GPS). NTP configuration is documented and monitored. Time synchronisation status is checked in the monthly monitoring report.",
        "PA": "Most servers are synchronised to an internal NTP server but some legacy systems and network devices have manual time settings that drift over time, causing log correlation issues.",
        "NA": "No clock synchronisation policy or NTP infrastructure exists. System clocks are set manually and drift over time.",
    },

    # ── T5: Access Control ────────────────────────────────────────────────────
    "T5.1.1": {
        "FA": "An Access Control Policy defines principles for access management: need-to-know, least privilege, segregation of duties, and formal authorisation. The policy covers all access types (logical, physical, privileged). Policy exceptions require documented approval. The policy is reviewed annually.",
        "PA": "Access control principles are referenced in the Information Security Policy, but a standalone Access Control Policy does not exist. Need-to-know and least privilege are applied informally rather than as documented requirements.",
        "NA": "No Access Control Policy exists.",
    },
    "T5.2.1": {
        "FA": "A formal user registration process requires manager authorisation for all new accounts. User IDs are unique and tied to a specific individual. Account creation is automated through an identity management system with approval workflows. User registration records are maintained for audit purposes.",
        "PA": "New user accounts require IT approval but manager authorisation is informally communicated rather than formally documented. Shared user IDs are used for some system functions, making individual accountability impossible.",
        "NA": "No formal user registration process exists. Accounts are created on request without documented approval.",
    },
    "T5.2.2": {
        "FA": "Privileged access (admin, root, DBA) is restricted to personnel with a documented business need. Privileged accounts are separate from regular user accounts. Privileged access is reviewed quarterly and revoked when no longer required. All privileged actions are logged.",
        "PA": "Privileged accounts are issued to IT staff but the number of privileged users is greater than required (24 admin accounts for a 5-person team). Quarterly reviews are not conducted and some accounts belong to former staff.",
        "NA": "No controls over privileged access exist. Multiple staff have admin accounts without documented justification.",
    },
    "T5.2.3": {
        "FA": "User security credentials (passwords, tokens, certificates) are managed according to a credentials management policy. Password complexity and length requirements are enforced technically. Initial passwords are unique per user and must be changed on first login. Credentials are distributed securely.",
        "PA": "Password complexity is enforced by Active Directory policy, but initial passwords are issued as a standard format (name + date of birth) that users often do not change promptly. Credentials for service accounts are not managed under the same policy.",
        "NA": "No credentials management policy or technical enforcement exists. Users choose their own passwords without complexity requirements.",
    },
    "T5.2.4": {
        "FA": "User access rights are reviewed quarterly for all systems. Reviews are conducted by system owners with manager sign-off. Excessive or inappropriate access is revoked within 5 business days of the review. Review records are maintained for audit. Privileged access undergoes monthly review.",
        "PA": "Access reviews are conducted annually for most systems but some systems (legacy applications, cloud platforms) are not included. Removal of access identified during reviews can take up to 30 days.",
        "NA": "No periodic user access reviews are conducted. Accounts accumulate access rights over time and are never cleaned up.",
    },
    "T5.3.1": {
        "FA": "A password policy mandates: minimum 12 characters, complexity requirements, 90-day maximum age for privileged accounts, prohibition on reuse of last 12 passwords, and account lockout after 5 failed attempts. MFA is required for all remote access and privileged operations.",
        "PA": "A password policy exists but MFA is only required for VPN access and not for privileged operations on internal systems. The 90-day password rotation requirement causes users to make minimal changes (incrementing a number).",
        "NA": "No password policy is in place. Password requirements are set to system defaults.",
    },
    "T5.4.1": {
        "FA": "A policy on use of network services restricts users to approved network services only. Unapproved use of cloud file sharing, VPN bypass tools, and peer-to-peer networks is technically blocked and policy-prohibited. Users must request access to new network services through a formal approval process.",
        "PA": "The network is filtered to block known malicious sites but use of unapproved cloud services and consumer-grade file sharing is not blocked. No formal approval process for new network services exists.",
        "NA": "No policy or controls on network service use are in place.",
    },
    "T5.4.2": {
        "FA": "Remote access requires MFA in addition to password authentication. VPN connections use certificate-based authentication. Remote user activity is monitored and logged. Inactive remote sessions are terminated after 15 minutes. Split tunnelling is disabled to ensure all traffic passes through corporate monitoring.",
        "PA": "Remote access via VPN uses password-only authentication. MFA is planned but not yet deployed. Remote sessions are not monitored and inactive sessions are not automatically terminated.",
        "NA": "No authentication controls for external connections exist beyond basic credentials.",
    },
    "T5.4.3": {
        "FA": "Equipment identification (MAC-based authentication via 802.1X or NAC) prevents unauthorised devices from connecting to the corporate network. A device registry is maintained. Unregistered devices connecting to the network generate security alerts.",
        "PA": "Network access control is applied in some areas (Wi-Fi, data centre) but not on general office network ports. Unregistered devices can connect to office network switches without authentication.",
        "NA": "No equipment identification controls exist. Any device can connect to the corporate network.",
    },
    "T5.4.4": {
        "FA": "Remote diagnostic and configuration ports (console ports, remote management interfaces) are disabled unless actively in use. When use is required, access is restricted to named individuals via firewall rules, and sessions are logged. Default vendor credentials are changed before deployment.",
        "PA": "Remote management access is restricted by firewall to IT subnets but remote diagnostic ports are not disabled when not in use. Default credentials have been changed on servers but not consistently on network devices.",
        "NA": "Remote management ports and diagnostic interfaces are open and accessible without restrictions.",
    },
    "T5.4.5": {
        "FA": "Network connection controls enforce access policies between network zones. Firewall rules are reviewed quarterly. Network connections from users are restricted to required services and protocols. Unused firewall rules are removed. Rule exceptions require documented approval.",
        "PA": "Firewall rules control access between the DMZ and internal network but internal zone-to-zone traffic is largely unrestricted. Firewall rule reviews are conducted annually rather than quarterly.",
        "NA": "No network connection controls exist between internal zones.",
    },
    "T5.4.6": {
        "FA": "Network routing is controlled to ensure traffic follows defined paths between security zones. Routing configurations are documented and reviewed quarterly. Routing changes go through the change management process. Route injection attacks are mitigated through routing protocol authentication.",
        "PA": "Core routing is configured appropriately but routing configuration management is informal. Changes to routing configurations are not always documented and quarterly reviews are not conducted.",
        "NA": "Network routing is not documented or actively managed.",
    },
    "T5.4.7": {
        "FA": "Wireless access uses WPA3-Enterprise with certificate-based authentication. Wireless networks are on separate VLANs with firewall-enforced access controls to internal systems. Rogue access point detection is deployed. Wireless security is reviewed annually.",
        "PA": "Corporate wireless uses WPA2-Enterprise but guest wireless shares the same authentication infrastructure. Rogue access point detection is not deployed.",
        "NA": "Wireless access uses WPA2-PSK with a shared password distributed broadly. No separation between corporate and guest wireless exists.",
    },
    "T5.5.1": {
        "FA": "Secure log-on procedures require unique user IDs, do not display last login information before authentication, limit login attempts to 5, do not identify which element (username or password) was incorrect, and require re-authentication after session timeout. Log-on procedures are enforced through Group Policy.",
        "PA": "Log-on procedures include username/password requirements but reveal whether the username or password was incorrect. Account lockout is configured for domain accounts but not for all application logins.",
        "NA": "No secure log-on procedures are enforced. Applications accept logins without lockout or error message controls.",
    },
    "T5.5.2": {
        "FA": "All users have unique identifiers. Shared accounts are prohibited except for specific documented exceptions with compensating controls. User identity is verified before account creation. Accounts are linked to individuals in the identity management system enabling full accountability.",
        "PA": "Individual user accounts are standard for most staff but several administrative and service accounts are shared among IT team members. Shared accounts do not have compensating logging controls.",
        "NA": "Shared accounts are common. Individual accountability for actions performed with shared credentials is not possible.",
    },
    "T5.5.3": {
        "FA": "The password management system enforces policy requirements automatically: complexity, minimum length, age, history, lockout. A self-service password reset portal is available with identity verification (MFA). Password management system health is monitored and alerts are configured for policy changes.",
        "PA": "Active Directory enforces password complexity and length, but self-service reset allows bypass of MFA verification in some scenarios. Password history is limited to last 5 passwords rather than 12.",
        "NA": "No password management system is in place. Password requirements are set per individual application without centralised enforcement.",
    },
    "T5.5.4": {
        "FA": "Use of system utilities (e.g., network scanners, password crackers, packet sniffers) is restricted to authorised staff and requires documented approval. Utility use is logged. Unapproved utilities are blocked on endpoints via application whitelisting.",
        "PA": "Use of system utilities requires informal manager approval but the policy is not technically enforced. Some utilities are available to all users through default software installations.",
        "NA": "No controls on system utility use exist. Any user can install and use system utilities.",
    },
    "T5.6.1": {
        "FA": "Information access is restricted based on classification and user role through role-based access control (RBAC). Access rights are provisioned based on business need and approved by the data owner. Data loss prevention (DLP) tools enforce classification-based access restrictions.",
        "PA": "Access to key systems is role-based, but access to shared drives and collaboration platforms is broad and not aligned with information classification. DLP is not deployed.",
        "NA": "No access restrictions on information exist. All information is accessible to all staff.",
    },
    "T5.6.2": {
        "FA": "Sensitive systems (payment processing, security management, HR data) are isolated from general user networks through dedicated network segments. Access is tightly controlled and monitored. System isolation requirements are documented and reviewed annually.",
        "PA": "The payment system is on a separate VLAN, but HR and security management systems are on the general server network accessible to any authorised user.",
        "NA": "No isolation of sensitive systems from general networks exists.",
    },
    "T5.6.3": {
        "FA": "A review process for publicly accessible content ensures that no sensitive or classified information is published. Content is approved by the information owner before publication. Regular automated scans and manual reviews check for accidental publication of sensitive data.",
        "PA": "A review process exists for the corporate website but not for all public channels (social media, industry portals). Automated scanning for sensitive content on public pages is not deployed.",
        "NA": "No review process for publicly accessible content exists.",
    },
    "T5.7.1": {
        "FA": "Mobile device access to corporate resources is controlled through MDM enrolment. Corporate data is containerised to prevent leakage to personal apps. Devices must meet security standards (OS version, encryption, screen lock). Lost/stolen device remote wipe is available and tested annually.",
        "PA": "Corporate email is available on personal mobile devices through ActiveSync, but MDM enrolment is not required. Remote wipe capability is limited to Exchange accounts and does not include other corporate data.",
        "NA": "No controls for mobile device access to corporate resources exist.",
    },
    "T5.7.2": {
        "FA": "Teleworking policy requires VPN use for all corporate system access, MFA authentication, use of organisation-issued equipment, and encrypted storage. Teleworkers receive specific security training. Policy compliance is monitored through VPN logs and endpoint management tools.",
        "PA": "A teleworking policy exists but allows use of personal equipment without encryption requirements. VPN use is recommended but not enforced, and some teleworkers access corporate systems directly.",
        "NA": "No teleworking policy or controls exist. Remote workers access corporate resources from personal devices without VPN.",
    },

    # ── T7: Application Security and Development ──────────────────────────────
    "T7.1.1": {
        "FA": "An Information Systems Acquisition, Development and Maintenance Policy addresses security requirements throughout the SDLC: from specification, through development and testing, to maintenance. The policy mandates security requirements analysis, secure coding standards, security testing, and code review. It is reviewed annually.",
        "PA": "Security is addressed in development guidelines but a formal SDLC security policy covering acquisition, development, and maintenance as a unified document does not exist.",
        "NA": "No SDLC security policy exists.",
    },
    "T7.2.1": {
        "FA": "Security requirements are specified at project inception and included in functional requirements documentation. A security requirements checklist covering OWASP top 10, authentication, authorisation, data validation, and encryption is used for all projects. Security requirements are reviewed and approved before development commences.",
        "PA": "Security requirements are raised during project initiation but are not formally documented in a security requirements specification. Coverage of security requirements varies by project and development team.",
        "NA": "Security requirements are not specified. Security is addressed reactively during or after development.",
    },
    "T7.2.2": {
        "FA": "Developer-provided software includes documented security training or materials for system administrators. Security configuration guidance, hardening checklists, and known vulnerability remediation instructions are provided with all internally developed and procured systems.",
        "PA": "Some vendor-provided systems include security configuration guides, but internally developed applications do not consistently include deployment security documentation.",
        "NA": "No security training or documentation is provided with systems.",
    },
    "T7.3.1": {
        "FA": "Input data validation is implemented on all application input fields. Validation checks are applied server-side (not just client-side). Validation rules are documented in the security requirements. Code reviews verify input validation implementation. Penetration testing includes injection attack testing.",
        "PA": "Input validation is implemented in most applications but relies on client-side validation that can be bypassed. Server-side validation is incomplete in some legacy applications. Code reviews do not consistently check validation logic.",
        "NA": "Input data validation is not implemented. Applications accept and process unvalidated data, creating SQL injection and XSS vulnerabilities.",
    },
    "T7.3.2": {
        "FA": "Internal processing integrity controls prevent unauthorised modifications to data during processing. Transaction logging captures before and after values for critical data changes. Reconciliation controls verify processing completeness. Integrity errors trigger alerts and investigation.",
        "PA": "Transaction logging exists for financial processes but integrity controls for non-financial data processing (e.g., user profile updates, access rights changes) are limited.",
        "NA": "No internal processing integrity controls exist. Data can be modified during processing without detection.",
    },
    "T7.3.3": {
        "FA": "Message integrity is protected using digital signatures or HMAC for all sensitive inter-system communications. Message authentication is implemented in APIs. Integrity verification failures are logged and trigger alerts. Cryptographic standards used for message authentication are reviewed annually.",
        "PA": "HTTPS is used for web application communications providing transport-level integrity, but message-level authentication (digital signatures, HMAC) is not implemented for internal API communications.",
        "NA": "No message integrity controls are implemented. Messages can be modified in transit without detection.",
    },
    "T7.3.4": {
        "FA": "Output data validation ensures that only appropriate and correctly classified data is presented to users. Output encoding prevents injection attacks. Data masking is applied to sensitive fields (e.g., displaying only last 4 digits of card numbers). Output validation is tested during security assessment.",
        "PA": "Some sensitive data (passwords, full card numbers) is masked in outputs, but comprehensive output validation is not consistently applied. XSS vulnerabilities caused by insufficient output encoding have been identified in penetration tests.",
        "NA": "No output data validation is implemented.",
    },
    "T7.4.1": {
        "FA": "A Cryptographic Controls Policy mandates approved algorithms (AES-256, RSA-2048/4096, SHA-256+), prohibits deprecated algorithms (DES, RC4, MD5, SHA-1 for new implementations), and defines key management requirements. Compliance is checked during code reviews and security assessments.",
        "PA": "Encryption is used for sensitive data but the specific algorithms and key lengths are not standardised across applications. Some applications use SHA-1 or weaker encryption that has not been prioritised for replacement.",
        "NA": "No cryptographic controls policy exists. Cryptographic decisions are made by individual developers.",
    },
    "T7.4.2": {
        "FA": "Cryptographic key management procedures cover the full key lifecycle: generation (using approved RNG), storage (HSM or encrypted key vault), distribution (secure channels), rotation (annually for data keys), and destruction (documented procedure). Key management is performed by designated key custodians.",
        "PA": "Keys are generated and stored but no formal key management procedure covers the full lifecycle. Key rotation is performed manually without a defined schedule. Key custodian roles are not formally assigned.",
        "NA": "No key management procedures exist. Cryptographic keys are stored in plaintext configuration files.",
    },
    "T7.5.1": {
        "FA": "Operational software is controlled through a change management process: new software or updates require risk assessment, testing in a non-production environment, approval, and documented rollback procedures. Unauthorised software installation is technically prevented on servers through application whitelisting.",
        "PA": "Software changes on production servers require change management approval, but application whitelisting is not deployed. Ad hoc fixes are sometimes applied directly to production without full change control.",
        "NA": "No controls over operational software installation or change exist.",
    },
    "T7.5.2": {
        "FA": "Production data used in testing must be anonymised or replaced with synthetic data before use. A data sanitisation procedure is documented and applied. Use of production data in testing without sanitisation requires CISO approval. Test data containing personal data is protected under data handling rules.",
        "PA": "Some test teams use anonymised data, but there is no organisation-wide policy or enforcement mechanism. Several teams still use production database snapshots directly in test environments.",
        "NA": "Production data is used in test environments without anonymisation or approval.",
    },
    "T7.5.3": {
        "FA": "Access to source code repositories is restricted to developers with a documented need. Read access and write access are separately controlled. All code commits are attributable to individual developers. Source code access is reviewed quarterly. External developers access code only through controlled integration environments.",
        "PA": "Source code is in a version-controlled repository with access controls but the number of users with write access exceeds operational requirements. Access reviews are not conducted.",
        "NA": "Source code is accessible to all IT staff without access controls.",
    },
    "T7.6.1": {
        "FA": "Software change control procedures require formal change request, impact assessment including security impact, testing (functional and security), approval by a change advisory board, and post-deployment verification. Emergency changes have an expedited process with retrospective review.",
        "PA": "Change control is applied to major releases but small bug fixes and patches are deployed with informal approval. Security impact assessment is not a standard step in the change control process.",
        "NA": "No software change control procedures exist.",
    },
    "T7.6.2": {
        "FA": "When operating systems are updated, all applications are reviewed for compatibility and potential security impact. A formal review process is conducted before OS update deployment. Application owners are notified and required to test their applications. Issues identified are addressed before OS change is deployed to production.",
        "PA": "OS updates are tested against core applications before deployment, but not all applications are systematically reviewed. Some application compatibility issues have been discovered after OS updates were applied to production.",
        "NA": "Operating system changes are made without reviewing the impact on application security.",
    },
    "T7.6.3": {
        "FA": "Modifications to commercial software packages are minimised and governed by a documented policy. Modifications require security impact assessment, approval, and are tested before deployment. Modifications are tracked to ensure they are reapplied after vendor updates. Modification rationale is documented.",
        "PA": "Software package modifications are documented but not assessed for security impact. Modified packages are sometimes overwritten during vendor updates, causing functionality issues.",
        "NA": "Software packages are modified ad hoc without documentation, controls, or consideration of security impact.",
    },
    "T7.6.4": {
        "FA": "Information leakage prevention controls include: DLP on email and web uploads, code review for hardcoded credentials or sensitive data exposure, network traffic monitoring for unusual outbound data volumes, and developer training on secure coding to prevent accidental leakage.",
        "PA": "DLP is deployed for email but not for web traffic or cloud uploads. Code reviews occur but do not have a checklist item for sensitive data exposure. Outbound traffic monitoring is limited.",
        "NA": "No information leakage prevention controls are in place.",
    },
    "T7.6.5": {
        "FA": "Outsourced software development is governed by contracts requiring: secure development practices (OWASP ASVS compliance), code reviews, security testing before delivery, source code escrow, intellectual property protection, and right-to-audit. Deliverables are independently security-tested before acceptance.",
        "PA": "Outsourced development contracts include confidentiality clauses but do not specify secure development standards or testing requirements. Code delivered is accepted on functional grounds without independent security testing.",
        "NA": "Outsourced software development occurs without security requirements, testing, or governance.",
    },
    "T7.7.1": {
        "FA": "A vulnerability management programme identifies, assesses, and remediates technical vulnerabilities in a timely manner. Vulnerability scans are conducted weekly for internet-facing systems and monthly for internal systems. Patch SLAs are defined: critical - 72 hours, high - 30 days, medium - 90 days. Patch status is reported monthly.",
        "PA": "Vulnerability scans are conducted quarterly for servers but internet-facing systems and applications are not scanned separately. Patch SLAs exist but are not consistently met for medium-priority vulnerabilities.",
        "NA": "No vulnerability management programme exists. Systems are patched reactively when vendor advisories are circulated.",
    },
    "T7.8.1": {
        "FA": "A supply chain security strategy addresses risks from software and hardware components sourced from third parties. The strategy covers supplier assessment, software bill of materials (SBOM) requirements, and controls for high-risk components. The strategy is reviewed annually and aligned with the organisation's risk management framework.",
        "PA": "Third-party software is reviewed for security during procurement but there is no comprehensive supply chain security strategy. SBOM requirements are not applied and high-risk component risks are assessed informally.",
        "NA": "No supply chain security strategy exists.",
    },
    "T7.8.2": {
        "FA": "Suppliers providing software or hardware components are periodically reviewed for security practices. Annual questionnaire-based assessments and triennial on-site audits are conducted for critical suppliers. Supplier security ratings are maintained in the vendor risk register.",
        "PA": "Supplier reviews occur at contract renewal (typically every 3 years) but not on an annual basis. Reviews focus on commercial terms and rarely address security practices in depth.",
        "NA": "No ongoing supplier security reviews are conducted after initial contract award.",
    },
    "T7.8.3": {
        "FA": "Controls to limit harm from compromised supply chain components include: network segmentation to isolate supplier-managed systems, code signing verification before deployment, monitoring for unexpected component behaviour, and incident response procedures specifically addressing supply chain compromise.",
        "PA": "Some controls exist to limit supply chain risk (code signing for approved software) but network isolation of supplier-managed systems and incident response procedures specific to supply chain compromise are not in place.",
        "NA": "No controls to limit harm from supply chain compromise exist.",
    },
    "T7.8.4": {
        "FA": "Supply chain operations security requirements are embedded in supplier contracts, covering: secure development practices, change notification obligations, security incident reporting (within 24 hours), and compliance with the organisation's information security requirements.",
        "PA": "Supplier contracts include general confidentiality terms but do not specify security requirements for supply chain operations, change notifications, or incident reporting timelines.",
        "NA": "Supply chain operations security requirements are not included in supplier contracts.",
    },
    "T7.8.5": {
        "FA": "Reliable delivery of critical components is ensured through: vetting of logistics providers, tamper-evident packaging verification on receipt, component authenticity verification, and alternate supplier arrangements for critical items. Delivery failures or anomalies are investigated under the incident management process.",
        "PA": "Key hardware is procured from approved vendors but tamper-evident packaging verification is not consistently performed on receipt. No alternate supplier arrangements exist for critical components.",
        "NA": "No controls for reliable and secure delivery of supply chain components exist.",
    },
    "T7.8.6": {
        "FA": "A process for identifying and addressing weaknesses in supply chain security includes: regular threat intelligence monitoring for reported supplier vulnerabilities, a defined process for receiving and acting on supplier security advisories, and contractual obligations for suppliers to proactively notify of identified weaknesses.",
        "PA": "Supplier security advisories are monitored for major vendors but no formal process exists for receiving, assessing, and acting on supply chain weaknesses in a timely manner.",
        "NA": "No process for addressing supply chain weaknesses or vulnerabilities exists.",
    },
    "T7.8.7": {
        "FA": "A register of critical information system components identifies single points of failure in the supply chain. For each critical component, alternate sources, stockpiling arrangements, or component substitution plans are documented. The register is reviewed annually and component criticality is assessed.",
        "PA": "Critical systems are identified in the business impact analysis but the supply chain components that comprise them are not analysed for single-source risk. No alternate supplier arrangements are in place.",
        "NA": "No assessment or management of supply chain risks for critical system components exists.",
    },

    # ── T9: Business Continuity ───────────────────────────────────────────────
    "T9.1.1": {
        "FA": "An Information Systems Continuity Management Policy is approved by management and defines requirements for continuity planning, testing, and maintenance. The policy covers BCM objectives, scope, roles and responsibilities, and links to the organisation's overall business continuity framework. It is reviewed annually.",
        "PA": "Business continuity requirements are addressed in a general BCM policy that mentions IT systems, but an IT-specific continuity management policy does not exist. IT continuity requirements are not formally derived from business impact analysis.",
        "NA": "No information systems continuity management policy exists.",
    },
    "T9.2.1": {
        "FA": "Information systems continuity plans are developed for all critical systems based on business impact analysis. Plans cover: recovery objectives (RTO/RPO), recovery procedures, roles and responsibilities, communication plans, and dependencies. Plans are approved by the business owner and stored securely off-site.",
        "PA": "A disaster recovery plan exists for the data centre covering infrastructure rebuild, but application-specific recovery procedures are incomplete. Recovery objectives are defined for some systems but are not formally agreed with business stakeholders.",
        "NA": "No information systems continuity plans exist. Recovery from a major outage would be entirely ad hoc.",
    },
    "T9.2.2": {
        "FA": "Information systems continuity plans are implemented and operational. Backup systems (hot/warm standby, geographic redundancy) are deployed for critical systems. Recovery tooling is tested and ready. Plan components (runbooks, contact lists, recovery scripts) are maintained and current.",
        "PA": "Backup systems exist for some critical applications (file storage, email) but not for all systems identified in the BIA. Recovery runbooks are outdated and do not reflect current system architecture.",
        "NA": "Continuity plans exist on paper but have never been operationally implemented. Backup systems are not in place.",
    },
    "T9.3.1": {
        "FA": "Information systems continuity plans are tested at least annually. Testing methods include: tabletop exercises (annually), component recovery tests (quarterly for backups), and full failover tests (every two years). Test results are documented, reviewed by management, and used to update plans. Identified gaps have defined remediation timelines.",
        "PA": "Annual disaster recovery drills are conducted for the primary data centre, but recovery of individual applications is not tested separately. Test results are documented but improvement actions are not consistently tracked.",
        "NA": "Continuity plans are never tested. Plan effectiveness is assumed but unverified.",
    },

    # ── M1 additions ─────────────────────────────────────────────────────────
    "M1.3.3": {
        "FA": "Contacts with relevant authorities (law enforcement, national CERT, regulators, emergency services) are documented and maintained. A procedure defines when and how to contact authorities, who is authorised to make contact, and how communications are documented. Contacts are reviewed and tested annually.",
        "PA": "Key authority contacts are held informally by senior staff but are not formally documented or regularly reviewed. There is no defined procedure for when to engage authorities or who is authorised to do so.",
        "NA": "No formal contacts with authorities are maintained. The organisation has no established relationship with law enforcement, CERT, or regulators for security matters.",
    },
    "M1.3.4": {
        "FA": "Memberships in special interest groups (ISACs, professional security associations, industry forums) are managed under a documented programme. Participation is approved by management, information sharing obligations are reviewed, and intelligence received is actioned and shared appropriately internally.",
        "PA": "Some staff participate in security forums but participation is informal and not managed or tracked by the organisation. Information received from these groups is not systematically disseminated.",
        "NA": "No participation in special interest groups or security communities exists.",
    },
    "M1.3.5": {
        "FA": "Risks related to external parties (customers, suppliers, partners, regulators) are formally identified and included in the risk register. External relationship risks are assessed before establishing new relationships and reviewed when significant changes occur. Contractual controls are derived from this risk assessment.",
        "PA": "External party risks are considered during contract negotiation for major relationships but are not systematically identified or maintained in the risk register. Risk assessment for external parties is not triggered by changes to existing relationships.",
        "NA": "No assessment of risks related to external parties is conducted.",
    },
    "M1.3.6": {
        "FA": "Security requirements for customer-facing systems and services are defined and implemented. Customer data is handled per data classification and privacy policies. Customers are informed of relevant security practices affecting their data. Security concerns raised by customers are tracked and addressed.",
        "PA": "Some security controls protect customer data but the organisation has not systematically identified and documented security requirements from the customer relationship perspective. Customer security inquiries are handled ad hoc.",
        "NA": "No security considerations are applied specifically to customer-facing systems or customer data handling beyond general IT security.",
    },
    "M1.3.7": {
        "FA": "Third-party agreements include a comprehensive set of information security requirements: data protection obligations, access controls, incident notification, audit rights, security standards compliance, and sub-processor restrictions. Security requirements are drafted with legal and security team input and reviewed at contract renewal.",
        "PA": "Third-party contracts include confidentiality and basic data protection clauses but specific security technical requirements, incident response timelines, and audit rights are not consistently included.",
        "NA": "Information security requirements are not included in third-party agreements.",
    },
    "M1.4.3": {
        "FA": "ISMS documentation is controlled: a document management system maintains current approved versions, a master list of ISMS documents is kept, obsolete documents are removed from use, and documents are reviewed on a defined schedule. Document changes are approved by the document owner.",
        "PA": "Key ISMS documents (policy, procedures) are version-controlled, but supplementary documents (guidelines, work instructions) are not consistently managed. Obsolete versions remain accessible on shared drives.",
        "NA": "ISMS documentation is not controlled. Multiple uncontrolled versions exist in different locations and it is unclear which version is current.",
    },

    # ── T6 additions ──────────────────────────────────────────────────────────
    "T6.2.3": {
        "FA": "Changes to third-party services are managed through a formal process requiring 30-day advance notice from the supplier. All changes affecting information security are reviewed and approved before implementation. Changes that exceed agreed service parameters require renegotiation. A change log is maintained.",
        "PA": "Third-party change notifications are received but reviewed informally. There is no formal approval step for changes affecting security and no change log for third-party service modifications.",
        "NA": "Third-party service changes are not tracked or managed. The organisation has no visibility into changes made by suppliers.",
    },
    "T6.3.2": {
        "FA": "Cloud service provider agreements include security provisions: data residency, encryption requirements, availability SLAs, security incident notification (within 24 hours), audit rights, data portability, and compliance certifications (ISO 27001, CSA STAR). Cloud security is reviewed annually.",
        "PA": "Cloud provider contracts include standard terms but negotiated security provisions (encryption requirements, audit rights, incident notification timelines) are not consistently included. Data residency requirements have not been formally assessed for all cloud services.",
        "NA": "Cloud providers are engaged without security-specific contractual requirements. No security review of cloud service agreements has been conducted.",
    },

    # ── T8 additions ──────────────────────────────────────────────────────────
    "T8.2.4": {
        "FA": "Incident response training is provided annually to all members of the Incident Response Team. Training covers technical response procedures, evidence handling, communication protocols, and regulatory reporting obligations. Training records are maintained and include practical exercises.",
        "PA": "Incident response team members receive general security training but dedicated incident response training covering all response phases is not provided. Practical exercises are not part of the training programme.",
        "NA": "No incident response training is provided to staff.",
    },
    "T8.2.5": {
        "FA": "Incident response capabilities are tested at least annually through tabletop exercises and biannually through simulation exercises. Test scenarios include data breaches, ransomware, insider threats, and DDoS. Test results are documented, reviewed by management, and used to update the Incident Response Plan.",
        "PA": "An annual tabletop exercise is conducted for major incident types but simulation exercises that test technical response capabilities are not performed. Exercise outcomes are not formally documented or used to update the plan.",
        "NA": "Incident response capabilities are never tested. Plan effectiveness is assumed but unverified.",
    },
    "T8.2.6": {
        "FA": "External incident response assistance is available through a pre-arranged contract with a specialist IR firm. The contract defines response time SLAs (on-site within 4 hours), scope of services, and data handling requirements. Internal staff know how to engage the external team and the engagement process is tested.",
        "PA": "External forensic and incident response support can be procured but there is no standing contract. Engaging external support during an active incident requires procurement approval, causing delays.",
        "NA": "No external incident response assistance arrangements exist.",
    },
    "T8.2.7": {
        "FA": "All information security incidents are documented in the incident management system with: incident description, timeline, affected systems, responders, actions taken, evidence collected, root cause, and closure notes. Documentation is retained for a minimum of three years. Templates ensure consistent documentation.",
        "PA": "Major incidents are documented but minor incidents are often closed with minimal notes. Documentation lacks consistent structure, and root cause information is frequently absent.",
        "NA": "Incidents are not formally documented. There is no incident record or tracking system.",
    },
    "T8.2.8": {
        "FA": "A lessons-learned process is applied after all significant incidents. A formal report is produced within 30 days covering root cause, impact, response effectiveness, and recommendations. Recommendations are tracked as action items. Annual trend analysis identifies systemic issues. Results are shared with relevant teams.",
        "PA": "Lessons-learned discussions occur informally after major incidents, but results are not formally documented or tracked. Recommendations are sometimes implemented but there is no mechanism to verify follow-through.",
        "NA": "No lessons-learned process exists. The same incident types recur without systematic improvement.",
    },
    "T8.2.9": {
        "FA": "Evidence collection procedures follow documented forensic guidelines: devices are imaged before analysis, chain of custody is maintained from collection to disposal, evidence is stored on write-protected media in a secure location, and evidence handling is logged. Staff involved in evidence collection have received forensic awareness training.",
        "PA": "Log files and screenshots are collected during incidents, but chain of custody documentation is not maintained. Evidence may be inadvertently modified by untrained responders before forensic analysis.",
        "NA": "No evidence collection procedures exist. Evidence is not preserved in a forensically sound manner.",
    },

    # ── T4: Communications and Network ──────────────────────────────────────
    "T4.1.1": {
        "FA": "The organisation has established a comprehensive Communications Policy that defines rules, standards, and requirements for all forms of information communication including internal and external channels. The policy covers acceptable use, classification of information during transmission, and responsibilities for all staff. It is reviewed annually and approved by senior management.",
        "PA": "The organisation has a general Acceptable Use Policy that includes some provisions about communications, such as prohibiting personal use of corporate email for sensitive matters. However, there is no dedicated Communications Policy addressing all transmission types, classification requirements, or formal responsibilities for information transfer across channels.",
        "NA": "No communications policy exists. Staff are expected to exercise personal judgment when transmitting information, and there are no formal rules governing the security of communications channels.",
    },
    "T4.2.1": {
        "FA": "The organisation has documented Information Transfer Procedures covering all transfer methods (email, USB, cloud, courier). Procedures specify encryption requirements, labelling, authorisation, and logging of transfers. All staff are trained on these procedures and compliance is monitored through periodic audits.",
        "PA": "The organisation has informal guidance on sending sensitive documents via email (e.g., use encryption for PII). Physical media transfer procedures and cloud-based transfer rules are not formally documented, and there is no central logging of information transfers.",
        "NA": "Information is transferred between parties without any formal procedures. Employees choose their own methods and tools for sharing files, with no encryption mandates or transfer logging in place.",
    },
    "T4.2.2": {
        "FA": "All third-party information-sharing arrangements are governed by formal Information Transfer Agreements. These agreements specify data classification, permitted transfer methods, encryption standards, retention rules, and breach notification obligations. Legal review is required before execution, and agreements are renewed annually.",
        "PA": "Non-disclosure agreements (NDAs) are in place with major partners, but they do not explicitly address technical controls for information transfer, permitted methods, or encryption requirements. Ad hoc sharing occurs without agreement updates.",
        "NA": "Information is shared with external parties on request without any formal agreements in place. There is no legal framework governing how external parties must protect transferred information.",
    },
    "T4.2.3": {
        "FA": "Physical media containing sensitive information is encrypted before transit and transported via approved secure couriers with tamper-evident packaging. Manifests are maintained and recipients must acknowledge receipt. Lost or damaged media is reported within 24 hours and investigated.",
        "PA": "Sensitive data on physical media is encrypted but courier selection is ad hoc and no formal chain-of-custody documentation is maintained. Receipts are not consistently obtained from recipients.",
        "NA": "Physical media is sent via standard post or handed to employees without encryption, tracking, or documentation of contents.",
    },
    "T4.2.4": {
        "FA": "The Electronic Messaging Policy requires all internal and external email containing sensitive information to be encrypted using organisational email encryption tools. Employees are prohibited from using personal email for business information. Messaging platforms are approved and reviewed for security before use.",
        "PA": "The organisation has rules about not sending passwords via email, but there is no comprehensive electronic messaging policy. Personal messaging apps are sometimes used for business discussions, and no technical controls enforce encryption.",
        "NA": "There are no controls on electronic messaging. Employees freely use personal email, SMS, and social messaging to share business information without restrictions.",
    },
    "T4.2.5": {
        "FA": "A registry of all business information systems and their integration points is maintained. Data flows between systems are documented and reviewed for security. Integration with external business systems requires security assessment and formal approval before connection.",
        "PA": "Core business systems are documented but integration with some third-party business platforms occurred without formal security review. Data flow documentation is incomplete for recently added systems.",
        "NA": "Business information systems are connected to external platforms on an ad hoc basis without documentation or security review of data flows.",
    },
    "T4.3.1": {
        "FA": "Electronic commerce transactions are protected by TLS 1.2+ encryption, and transaction integrity is verified using digital signatures. Customer payment data is handled in accordance with PCI-DSS requirements. Security of the e-commerce platform is tested annually by an external party.",
        "PA": "The organisation's e-commerce site uses HTTPS but digital signatures for transaction integrity are not implemented. PCI-DSS scope has been assessed but not all requirements are met, particularly around access controls to cardholder data.",
        "NA": "The organisation's online store transmits transaction data without encryption or integrity verification. No security assessments have been conducted on the e-commerce platform.",
    },
    "T4.3.2": {
        "FA": "Online transactions require multi-factor authentication, session tokens have a 15-minute timeout, and all transaction data is encrypted end-to-end. Anti-fraud controls including anomaly detection are deployed and reviewed monthly. Transaction logs are retained for 12 months.",
        "PA": "Online transactions use password authentication but MFA is not enforced for all customers. Session timeouts are configured but longer than policy recommends. Fraud monitoring exists but uses manual review rather than automated anomaly detection.",
        "NA": "Online transactions rely on single-factor authentication with no session management controls. No fraud detection is in place.",
    },
    "T4.3.3": {
        "FA": "All publicly available information is reviewed and approved by an authorised owner before publication. A web content governance policy defines what information may be published and includes a classification check. Content is reviewed annually for accuracy and continued appropriateness.",
        "PA": "A process exists for approving press releases before publication, but departmental websites can be updated by staff without security review. Sensitive internal information has occasionally appeared on public pages before being identified and removed.",
        "NA": "Employees can publish information to the corporate website without approval. There are no controls preventing accidental disclosure of sensitive internal information on public channels.",
    },
    "T4.4.1": {
        "FA": "Connection to information sharing platforms (ISACs, sector-specific threat intel communities) is governed by a formal registration process including legal review, data classification assessment, and CISO approval. Shared information is classified and sanitised before submission. Participation is reviewed annually.",
        "PA": "The organisation participates in an information sharing community but the connection was established informally. There is no documented policy for what information may be shared, and sharing decisions are made ad hoc by individual analysts.",
        "NA": "The organisation does not participate in any formal information sharing platforms and has no process for doing so.",
    },
    "T4.4.2": {
        "FA": "Before releasing information into information sharing communities, a formal review process is applied to verify classification, redact personal and sensitive data, and obtain management approval. A log of all releases is maintained with the rationale for each release.",
        "PA": "Threat indicators are shared with the ISAC community but the review process is informal. A technical team member decides what to share without formal approval or documentation, and no records are kept of what has been released.",
        "NA": "Information is released into sharing communities without review. Sensitive internal data including system configuration details has been included in shared indicators.",
    },
    "T4.5.1": {
        "FA": "Network controls are documented in the Network Security Standard. The standard mandates firewall deployment at all network boundaries, network segmentation between trust zones, and access control lists. Network configurations are reviewed quarterly and changes go through a formal change management process.",
        "PA": "Firewalls are deployed at the perimeter and some internal segmentation exists for the data centre. However, network security standards are not fully documented, and network access control lists are inconsistently applied across all segments.",
        "NA": "No formal network controls exist. The internal network is flat with no segmentation, and there are no documented standards for network configuration.",
    },
    "T4.5.2": {
        "FA": "All network services (VPN, DNS, DHCP, proxy) are included in the network services inventory with defined security requirements. Service level agreements and security terms are documented for managed network services. Security attributes of all third-party network services are reviewed before procurement.",
        "PA": "Core network services like VPN and firewall have documented security requirements, but cloud-based DNS and DHCP management services were procured without a formal security review. SLA security terms are not consistently included in network service contracts.",
        "NA": "Network services are used without documentation of their security attributes or formal assessment of their security capability.",
    },
    "T4.5.3": {
        "FA": "Network segmentation policy defines mandatory trust zones: Internet DMZ, user workstation network, server network, and restricted management network. Firewall rules enforce traffic restrictions between zones and are reviewed quarterly. VLAN implementation is documented and validated through penetration testing annually.",
        "PA": "A perimeter DMZ exists separating the Internet from the internal network. However, internal user workstations, servers, and management systems are on the same flat network with no further segmentation.",
        "NA": "All systems (workstations, servers, network devices) are on a single flat network with no segmentation or access controls between them.",
    },
    "T4.5.4": {
        "FA": "Wireless networks are segregated from the wired corporate network using separate VLANs. All wireless access points use WPA3-Enterprise with certificate-based authentication. Guest Wi-Fi is isolated with no access to internal systems. Wireless networks are scanned monthly for rogue access points.",
        "PA": "Corporate wireless uses WPA2-PSK with a shared key that is changed quarterly. The wireless network is on a separate VLAN but the SSID and pre-shared key are widely known. Guest Wi-Fi is provided but shares infrastructure with the corporate network.",
        "NA": "Wireless access points use WEP or open authentication with no network isolation. No controls exist to prevent unauthorised wireless access.",
    },

    # ── T8: Incident Management ──────────────────────────────────────────────
    "T8.1.1": {
        "FA": "An Information Security Incident Management Policy is approved by the CISO and reviewed annually. The policy defines roles and responsibilities, incident categories and severity levels, reporting obligations (including regulatory), and the lifecycle from detection to closure. All staff receive annual awareness training on the policy.",
        "PA": "An IT incident response procedure exists covering system outages and malware events, but it does not address the full scope of information security incidents (e.g., data breaches, insider threats) and is not formally approved by management or reviewed on a regular schedule.",
        "NA": "No incident management policy exists. Security incidents are handled ad hoc by the IT helpdesk alongside regular IT support requests.",
    },
    "T8.1.2": {
        "FA": "Responsibilities and procedures for incident management are documented in the Incident Response Playbook. Designated Incident Response Team members have defined roles, and contact lists are maintained and tested quarterly. Escalation procedures are documented for each incident severity level.",
        "PA": "An informal incident response team exists but roles are not formally assigned in writing. Some playbooks exist for major incident types but coverage is incomplete, and escalation paths are unclear for novel incident types.",
        "NA": "No formal incident response responsibilities or procedures are defined. Whoever is available responds to incidents without defined roles or steps to follow.",
    },
    "T8.1.3": {
        "FA": "Information security incidents are reported through a dedicated security ticketing system accessible to all employees. A 24-hour reporting SLA is enforced. Automated alerts from SIEM tools are integrated into the ticketing system. All reports are acknowledged within 1 hour.",
        "PA": "Employees can report security incidents via email to the IT helpdesk. However, there is no dedicated security reporting channel, reports are not always prioritised appropriately, and the SLA for acknowledgement is not consistently met.",
        "NA": "There is no formal mechanism for reporting security incidents. Employees report issues verbally to their manager, and there is no centralised tracking.",
    },
    "T8.2.1": {
        "FA": "The Incident Response Plan documents the full lifecycle: preparation, identification, containment, eradication, recovery, and post-incident review. The plan is tested via tabletop exercises twice yearly. Lessons learned are formally captured and used to update procedures within 30 days of an incident.",
        "PA": "An incident response plan exists covering containment and recovery steps for common scenarios. Post-incident reviews are not consistently conducted, and lessons learned are not formally tracked or used to update the plan.",
        "NA": "Incident response is handled reactively without a documented plan. There is no post-incident review process.",
    },
    "T8.2.2": {
        "FA": "Evidence collected during security incidents is handled under a documented evidence management procedure aligned with forensic best practices. Chain of custody is maintained, evidence is stored on write-protected media, and all handling is logged. External forensic support is available under pre-arranged contract.",
        "PA": "IT staff capture system logs and screenshots during incidents, but there is no formal evidence handling procedure. Chain-of-custody documentation is not maintained and evidence may be inadvertently modified.",
        "NA": "No evidence collection or preservation procedures exist. System logs are often overwritten before incident analysis can be completed.",
    },
    "T8.2.3": {
        "FA": "Incidents involving personal data are reported to the relevant data protection authority within 72 hours of discovery in accordance with applicable regulations. An internal escalation procedure ensures the privacy officer is notified within 4 hours. Template notifications are maintained and reviewed annually.",
        "PA": "The organisation is aware of regulatory reporting obligations for data breaches but the internal escalation path to the privacy officer is not clearly documented. Previous incidents have been reported late due to delays in escalation.",
        "NA": "No process exists for regulatory reporting of security incidents. Staff are not aware of reporting obligations.",
    },
    "T8.3.1": {
        "FA": "The Incident Response Plan includes a dedicated section on continuity during incidents. Key controls address maintaining critical services during containment, activating backup systems where primary systems are isolated, and communication with stakeholders. The plan is integrated with the Business Continuity Plan.",
        "PA": "The incident response plan references the business continuity plan but the two are not integrated. During major incidents, it is unclear who is responsible for activating business continuity measures.",
        "NA": "Incident response and business continuity are managed separately with no coordination. Critical services may be disrupted during security incidents with no fallback.",
    },
    "T8.3.2": {
        "FA": "Forensic analysis capabilities are available through a contracted specialist firm with documented SLAs for response times. Internal staff are trained in first-responder forensic procedures (preservation and documentation). The organisation participates in an annual forensic exercise.",
        "PA": "The organisation relies on its IT team for forensic analysis but staff have not received specialist training. External forensic support is available but is not under a pre-arranged contract, leading to delays in engagement during incidents.",
        "NA": "No forensic analysis capability exists, internal or external. Evidence collected during incidents is often inadmissible or incomplete.",
    },
    "T8.3.3": {
        "FA": "Post-incident reviews are mandatory for all incidents rated Severity 1 or 2. Reviews follow a formal template covering timeline, root cause, impact assessment, and recommendations. Action items are tracked in the risk register and reported to the CISO. Trend analysis is conducted quarterly.",
        "PA": "Post-incident reviews are conducted informally after major incidents but recommendations are not always tracked. Minor incidents are closed without review, missing opportunities to identify systemic issues.",
        "NA": "No post-incident review process exists. Incidents are closed when the immediate issue is resolved with no analysis of root cause or lessons learned.",
    },

    # ── M1: Strategy and Planning ────────────────────────────────────────────
    "M1.1.1": {
        "FA": "The organisation has conducted a formal context analysis exercise documenting all external and internal factors affecting information security. The analysis covers regulatory environment, business sector risks, internal capability assessments, and stakeholder requirements. It is reviewed and updated annually as part of the strategic planning cycle.",
        "PA": "A risk assessment covers some internal and external factors, but there is no systematic context analysis addressing all relevant stakeholder requirements, sector-specific risks, and internal capabilities as a unified exercise.",
        "NA": "No formal analysis of the organisation's context in relation to information security has been conducted. Information security arrangements have evolved reactively without consideration of external or internal factors.",
    },
    "M1.1.2": {
        "FA": "Top management formally approves the Information Security Policy annually and chairs the Information Security Steering Committee. Management commitment is evidenced by dedicated ISMS budget allocation, mandatory security training for all staff, and regular board-level reporting on information security performance.",
        "PA": "Senior management approved the Information Security Policy at inception, but ongoing involvement is limited. Security decisions are largely delegated to the IT department without senior management oversight or regular board-level reporting.",
        "NA": "Information security is managed entirely within the IT team without executive sponsorship. No management commitment statements, dedicated budgets, or board-level reporting on security exist.",
    },
    "M1.1.3": {
        "FA": "Roles and responsibilities for information security are defined in the ISMS documentation and the RACI matrix. Each role has a documented job description with security responsibilities. The CISO has organisation-wide authority for information security matters. Segregation of duties is enforced for critical security functions.",
        "PA": "The CISO role is defined and security responsibilities are assigned to key IT positions, but information security responsibilities for non-IT roles (e.g., data owners, process owners) are not formally documented or communicated.",
        "NA": "No formal assignment of information security roles and responsibilities exists. Security tasks are performed informally by whoever is available.",
    },
    "M1.1.4": {
        "FA": "An asset inventory is maintained with designated asset owners for all information assets. Owners are responsible for approving access rights, reviewing asset classifications, and ensuring appropriate protection measures are in place. Owner responsibilities are documented in the asset management policy.",
        "PA": "Asset owners are assigned for major systems but the inventory is incomplete, covering only IT-managed assets. Information assets held by business units often lack a designated owner, leading to inconsistent protection.",
        "NA": "Asset ownership is not formally assigned. Information assets are managed collectively by IT with no individual accountability for protection.",
    },
    "M1.2.1": {
        "FA": "An Information Security Policy approved by the board is in place and communicated to all employees. The policy is reviewed annually and after significant incidents or changes. All new employees receive the policy during onboarding, and annual re-acknowledgement is required.",
        "PA": "An Information Security Policy exists and has been emailed to all staff, but there is no process to ensure staff read or acknowledge it. The policy has not been reviewed since its initial publication three years ago.",
        "NA": "No formal Information Security Policy exists. Security expectations are communicated verbally by managers as needed.",
    },
    "M1.2.2": {
        "FA": "Information security objectives are documented and measurable, aligned with the organisation's strategic goals. Objectives cover confidentiality, integrity, and availability targets with defined KPIs. Progress against objectives is reported to management quarterly and results inform annual planning.",
        "PA": "High-level security objectives are referenced in the Information Security Policy, but they are not translated into measurable targets or KPIs. There is no mechanism to track or report progress against objectives.",
        "NA": "No information security objectives have been defined. Security activities are reactive rather than goal-driven.",
    },
    "M1.3.1": {
        "FA": "An information security risk assessment methodology is documented, approved, and applied consistently across the organisation. The methodology defines criteria for risk acceptance, likelihood and impact scales, and treatment options. Risk assessments are conducted annually and when significant changes occur.",
        "PA": "Risk assessments are conducted for major projects, but the methodology varies between teams and is not consistently applied. Risk acceptance criteria are not formally defined, and not all organisational areas undergo periodic assessment.",
        "NA": "No formal risk assessment process exists. Security investments are based on incident history rather than systematic risk evaluation.",
    },
    "M1.3.2": {
        "FA": "A risk treatment plan is maintained for all identified risks above the acceptance threshold. Treatment options (mitigate, accept, transfer, avoid) are documented with owners, timelines, and residual risk assessments. Plan progress is reviewed quarterly and approved risks are tracked until closure.",
        "PA": "High-severity risks are addressed with mitigation plans, but medium risks are often accepted without formal documentation. Risk owners are not consistently assigned and treatment plans lack defined timelines.",
        "NA": "Risks are identified but not formally treated. There is no risk treatment plan, and identified risks are not tracked to resolution.",
    },
    "M1.4.1": {
        "FA": "Adequate resources (budget, personnel, technology) for information security are allocated annually based on the risk assessment outcomes. Budget allocation is approved by executive management and reviewed mid-year. Staffing levels are benchmarked against industry norms.",
        "PA": "A security budget exists but is allocated to reactive spending (e.g., incident response) rather than proactive risk management. Resource allocation decisions are made without systematic needs analysis.",
        "NA": "No dedicated budget or staffing for information security exists. Security tasks are handled by IT staff alongside their primary responsibilities.",
    },
    "M1.4.2": {
        "FA": "Information security competency requirements are defined for all roles with security responsibilities. Competency assessments are conducted annually and gaps are addressed through a training plan. Specialist certifications (CISSP, CISM, etc.) are maintained for key security roles.",
        "PA": "The CISO and security team members have relevant certifications but competency requirements are not formally defined for other roles with security responsibilities (e.g., system administrators, data owners).",
        "NA": "No competency requirements for information security are defined. Staff are expected to self-develop security skills.",
    },
    "M1.5.1": {
        "FA": "Communication procedures ensure that security-relevant information is distributed to appropriate personnel in a timely manner. Escalation paths, notification templates, and communication channels (including during incidents) are documented. Communication plans are tested annually.",
        "PA": "Security alerts and policy updates are communicated via email to IT staff, but there is no formal communication procedure ensuring the right information reaches the right people in a timely manner.",
        "NA": "Security information is shared informally within the IT team with no structured communication procedures or escalation paths.",
    },
    "M1.6.1": {
        "FA": "A formal internal audit programme covering the ISMS is conducted annually by qualified internal auditors independent of the operations under review. Audit findings are tracked to closure in a register. Results are reported to top management and the audit committee.",
        "PA": "IT audits are conducted as part of the organisation's broader internal audit programme, but dedicated ISMS audits are not planned. Security controls are only reviewed when they fall within the scope of a business process audit.",
        "NA": "No internal audit of information security controls is conducted.",
    },
    "M1.7.1": {
        "FA": "Management reviews the ISMS at least annually covering: changes in context, risk assessment results, audit findings, performance metrics, and stakeholder feedback. Review outcomes are recorded in management review minutes and trigger updates to the ISMS where necessary.",
        "PA": "An annual review of the security policy occurs, but it is not a comprehensive ISMS management review. Security performance metrics and risk treatment status are not formally reviewed at this meeting.",
        "NA": "No management review of the ISMS is conducted.",
    },

    # ── T6: Third-Party Security ──────────────────────────────────────────────
    "T6.1.1": {
        "FA": "A Third-Party Security Policy defines requirements for all supplier and partner relationships involving access to information assets. The policy covers risk classification, due diligence requirements, contractual obligations, and ongoing monitoring. All third-party relationships are categorised by risk tier.",
        "PA": "Supplier security requirements are addressed within contracts but there is no overarching Third-Party Security Policy. Requirements are inconsistently applied across different procurement teams, and risk-based tiering of suppliers is not formalised.",
        "NA": "No policy or formal requirements exist for the security of third-party relationships. Suppliers are engaged without security assessment or contractual security obligations.",
    },
    "T6.1.2": {
        "FA": "Third-party security requirements are documented in a standard contract schedule and reviewed by legal and security teams before signature. Requirements cover data handling, access controls, incident notification (within 24 hours), audit rights, and sub-processor obligations. Contracts are renewed with updated requirements annually.",
        "PA": "Standard contract templates include an NDA and general data protection clauses, but specific information security requirements (e.g., encryption standards, incident response timelines, audit rights) are not consistently included. Incident notification obligations are vague.",
        "NA": "Supplier contracts do not contain information security requirements. Data handling and security obligations are not defined contractually.",
    },
    "T6.1.3": {
        "FA": "The supply chain security programme includes assessment of third parties' security practices via questionnaires, certifications review (e.g., ISO 27001, SOC 2), and on-site audits for critical suppliers. Supply chain risks are documented in the risk register. Critical suppliers undergo annual reassessment.",
        "PA": "High-risk suppliers complete a security questionnaire during onboarding but ongoing monitoring is limited. Sub-contractor security arrangements (fourth-party risk) are not assessed.",
        "NA": "No assessment of supply chain security risks is performed. Third parties' security practices are assumed to be adequate without verification.",
    },
    "T6.2.1": {
        "FA": "Third-party access to systems and information is governed by a formal access approval process. Access is granted on a least-privilege basis, time-limited, and revoked immediately upon contract termination. All third-party access is logged and reviewed quarterly.",
        "PA": "Third-party access is managed but the process is inconsistent. Some legacy supplier accounts remain active after contract termination, and access reviews are not conducted systematically.",
        "NA": "Third parties are given access to systems as required by the business with no formal approval process, and access is not systematically revoked when the relationship ends.",
    },
    "T6.2.2": {
        "FA": "Third-party service delivery is monitored against defined security SLAs. Monthly reports are reviewed by the security team, and non-conformances are formally managed. Annual security audits of critical suppliers are conducted using the contractually reserved audit right.",
        "PA": "Suppliers provide service reports but security-specific performance metrics are not tracked. Audit rights are included in contracts but have not been exercised. Supplier non-conformances are handled informally.",
        "NA": "Third-party service delivery is monitored for operational performance only. No security monitoring or assessment of third-party delivery is conducted.",
    },
    "T6.3.1": {
        "FA": "Changes to third-party services are managed through the change management process with a security impact assessment required for significant changes. Third parties must notify the organisation of changes affecting security at least 30 days in advance. Change notifications are tracked and reviewed by the security team.",
        "PA": "The change management process covers internal changes but third-party change notifications are handled informally. There is no requirement for advance notice of supplier changes that might affect information security.",
        "NA": "Third-party service changes are not tracked. The organisation has discovered critical supplier changes (e.g., data centre relocation, platform migrations) after they occurred.",
    },
}


# ── Fallback clause generator ─────────────────────────────────────────────────
def make_fallback(control_id: str, control_name: str, family: str, desc: str) -> dict:
    """Generate generic FA/PA/NA clauses based on control structure."""
    subject = control_name.replace(" and ", " & ")
    return {
        "FA": (
            f"The organisation has implemented comprehensive controls for {subject} in compliance "
            f"with {family} requirements. A formal policy is approved by management, communicated to all "
            f"relevant staff, and reviewed annually. Controls are documented, applied consistently, and "
            f"verified through periodic audits. Evidence of effectiveness is maintained and reported to "
            f"management."
        ),
        "PA": (
            f"Some controls exist for {subject} but implementation is incomplete. A policy or procedure "
            f"has been drafted but not formally approved or consistently applied across all areas. Key "
            f"requirements of the {family} control are addressed partially, but gaps remain in either "
            f"documentation, enforcement, or monitoring."
        ),
        "NA": (
            f"No controls are in place for {subject}. The organisation has not implemented the "
            f"requirements of this {family} control. No policy, procedure, or technical measure "
            f"addresses this area, and staff are not aware of the associated security requirements."
        ),
    }


def load_controls():
    path = ROOT / "data/02_processed/uae_ia_controls_clean.json"
    data = json.load(open(path))
    seen = set()
    controls = []
    for c in data:
        cid = c["control"]["id"]
        if cid in seen:
            continue
        seen.add(cid)
        desc = c["control"]["description"]
        name = c["control"]["name"].split(" - ", 1)[-1]
        family = cid.split(".")[0]
        # text_snippet: first 300 chars of description
        snippet = desc[:300].replace("\n", " ").strip()
        controls.append(
            {
                "id": cid,
                "name": name,
                "family": family,
                "snippet": snippet,
            }
        )
    return controls


def build_records(controls):
    records = []
    for c in controls:
        cid = c["id"]
        clauses = CLAUSES.get(cid) or make_fallback(cid, c["name"], c["family"], c["snippet"])
        for label, policy_text in [
            ("Fully Addressed", clauses["FA"]),
            ("Partially Addressed", clauses["PA"]),
            ("Not Addressed", clauses["NA"]),
        ]:
            short = {"Fully Addressed": "FA", "Partially Addressed": "PA", "Not Addressed": "NA"}[label]
            records.append(
                {
                    "control_id": cid,
                    "corrected_control_id": cid,
                    "control_name": c["name"],
                    "control_text_snippet": c["snippet"],
                    "policy_passage_id": f"synthetic_{cid}_{short}",
                    "policy_name": "synthetic",
                    "policy_section": "synthetic",
                    "compliance_status": label,
                    "confidence": 4,
                    "is_hard_negative": False,
                    "mismatch_reason": "synthetic",
                    "evidence_or_notes": "synthetically generated",
                    "comments": "auto-generated",
                    "policy_text_snippet": policy_text,
                }
            )
    return records


def main():
    print("Loading controls...")
    controls = load_controls()
    print(f"  Total: {len(controls)} controls")

    families = {}
    for c in controls:
        families.setdefault(c["family"], []).append(c)

    print("Generating records...")
    all_records = []
    for family in sorted(families):
        fam_controls = families[family]
        recs = build_records(fam_controls)
        all_records.extend(recs)
        fa = sum(1 for r in recs if r["compliance_status"] == "Fully Addressed")
        pa = sum(1 for r in recs if r["compliance_status"] == "Partially Addressed")
        na = sum(1 for r in recs if r["compliance_status"] == "Not Addressed")
        print(f"  [{family}] {len(fam_controls)} controls → FA={fa} PA={pa} NA={na}")

    # Save
    out_path = OUT_DIR / "synthetic_training_data.json"
    json.dump(all_records, open(out_path, "w"), indent=2)
    print(f"\nSaved {len(all_records)} records → {out_path}")

    # Print coverage stats
    in_clauses = sum(1 for c in controls if c["id"] in CLAUSES)
    fallback = len(controls) - in_clauses
    print(f"  Custom clauses: {in_clauses}/{len(controls)} controls")
    print(f"  Fallback generated: {fallback}/{len(controls)} controls")


if __name__ == "__main__":
    main()
