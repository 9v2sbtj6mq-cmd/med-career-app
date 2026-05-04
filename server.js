const express = require("express");
const dotenv = require("dotenv");
const mammoth = require("mammoth");
const { tavily } = require("@tavily/core");
const { chromium } = require("playwright");
const NodeCache = require("node-cache");
const { Document, Packer, Paragraph, HeadingLevel, TextRun, AlignmentType } = require("docx");

dotenv.config();

const app = express();
app.use(express.json({ limit: "25mb" }));
app.use(express.static("."));

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const TAVILY_API_KEY = process.env.TAVILY_API_KEY;
const FIRECRAWL_API_KEY = process.env.FIRECRAWL_API_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const MODEL_FAST = process.env.GEMINI_MODEL_FAST || "gemini-2.5-flash-lite";
const MODEL_SMART = process.env.GEMINI_MODEL_SMART || "gemini-2.5-flash";
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL || "meta-llama/llama-3.1-8b-instruct:free";
const SCORING_PROVIDER = (process.env.SCORING_PROVIDER || "groq").toLowerCase();
const FIRECRAWL_TOP_N = Number(process.env.FIRECRAWL_TOP_N || 10);

const tavilyClient = tavily({ apiKey: TAVILY_API_KEY });
const searchCache = new NodeCache({ stdTTL: 900 });
const scoreCache = new NodeCache({ stdTTL: 1800 });
const geminiCache = new NodeCache({ stdTTL: 3600 });

async function askGemini(prompt, model = MODEL_SMART) {
  if (!GOOGLE_API_KEY) throw new Error("Missing GOOGLE_API_KEY in .env file.");

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GOOGLE_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
    }
  );

  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "Gemini API error");

  return data.candidates?.[0]?.content?.parts?.map(p => p.text).join("\n") || "";
}

async function askGroq(prompt, model = GROQ_MODEL) {
  if (!GROQ_API_KEY) throw new Error("Missing GROQ_API_KEY in .env file.");

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content: "You are a strict JSON-only Australian medical recruitment scoring assistant. Return valid JSON only."
        },
        {
          role: "user",
          content: prompt
        }
      ]
    })
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "Groq API error");

  return data.choices?.[0]?.message?.content || "";
}

async function askOpenRouter(prompt, model = OPENROUTER_MODEL) {
  if (!OPENROUTER_API_KEY) throw new Error("Missing OPENROUTER_API_KEY in .env file.");

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "Med Career App"
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content: "You are a strict JSON-only Australian medical recruitment scoring assistant. Return valid JSON only."
        },
        {
          role: "user",
          content: prompt
        }
      ]
    })
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || "OpenRouter API error");

  return data.choices?.[0]?.message?.content || "";
}

async function askScoringModel(prompt) {
  try {
    if (SCORING_PROVIDER === "openrouter") {
      return await askOpenRouter(prompt);
    }

    if (SCORING_PROVIDER === "gemini") {
      return await askGemini(prompt, MODEL_FAST);
    }

    return await askGroq(prompt);
  } catch (primaryError) {
    console.warn(`${SCORING_PROVIDER} scoring failed. Falling back to Gemini fast model:`, primaryError.message);
    return askGemini(prompt, MODEL_FAST);
  }
}

async function scrapeJobWithFirecrawl(url) {
  if (!FIRECRAWL_API_KEY) return "";
  if (!url || !/^https?:\/\//i.test(url)) return "";

  try {
    const response = await fetch("https://api.firecrawl.dev/v1/scrape", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${FIRECRAWL_API_KEY}`
      },
      body: JSON.stringify({
        url,
        formats: ["markdown"],
        onlyMainContent: true,
        waitFor: 1000
      })
    });

    const data = await response.json();
    if (!response.ok) {
      console.warn("Firecrawl scrape failed:", data.error || data.message || response.statusText);
      return "";
    }

    return data.data?.markdown || data.markdown || "";
  } catch (error) {
    console.warn("Firecrawl scrape error:", error.message);
    return "";
  }
}

async function enrichTopJobsWithFirecrawl(jobs) {
  const topJobs = jobs.slice(0, FIRECRAWL_TOP_N);
  const remainingJobs = jobs.slice(FIRECRAWL_TOP_N);

  const enrichedTopJobs = await Promise.all(
    topJobs.map(async job => {
      const fullDescription = await scrapeJobWithFirecrawl(job.link);

      return addClosingDateInfo({
        ...job,
        fullDescription: fullDescription ? fullDescription.slice(0, 12000) : job.fullDescription || "",
        descriptionSource: fullDescription ? "firecrawl" : job.descriptionSource || "snippet"
      });
    })
  );

  return [...enrichedTopJobs, ...remainingJobs];
}

async function askGeminiCached(cacheKey, prompt, model = MODEL_SMART) {
  const cached = geminiCache.get(cacheKey);

  if (cached) {
    return cached;
  }

  const result = await askGemini(prompt, model);
  geminiCache.set(cacheKey, result);
  return result;
}

function extractJson(text) {
  const cleaned = text.replace(/```json/g, "").replace(/```/g, "").trim();
  const start = cleaned.indexOf("[");
  const end = cleaned.lastIndexOf("]");
  if (start === -1 || end === -1) return [];

  try {
    return JSON.parse(cleaned.slice(start, end + 1));
  } catch {
    return [];
  }
}

function parseScore(scoreText) {
  const match = String(scoreText || "").match(/Score:\s*([0-5](?:\.\d+)?)/i);
  return match ? Number(match[1]) : 0;
}

function buildScoreText(score) {
  if (!score) {
    return "Score not available. Try again or paste full job description.";
  }

  const breakdown = score.breakdown || {};
  const numericScore = Number(score.score || 0);

  return [
    `Score: ${numericScore}/5`,
    `Recommendation: ${score.recommendation || "Maybe"}`,
    `Confidence: ${score.confidence || "Medium"}`,
    `Apply readiness: ${score.applyReadiness || "Needs manual check"}`,
    "",
    "Breakdown:",
    `Registration: ${breakdown.registration ?? "?"}/1`,
    `Visa: ${breakdown.visa ?? "?"}/1`,
    `Level: ${breakdown.level ?? "?"}/1`,
    `Training value: ${breakdown.trainingValue ?? "?"}/1`,
    `Practical fit: ${breakdown.practicalFit ?? "?"}/1`,
    "",
    "Reason:",
    score.reason || "No reason provided.",
    "",
    "Warning:",
    score.warning || "No major warning identified from supplied information."
  ].join("\n");
}

function makeGeminiCacheKey(prefix, profile, job, model = "") {
  const raw = `${prefix}:${model}:${profile || ""}:${job || ""}`;
  return raw.slice(0, 5000);
}


function profileToText(profile) {
  if (!profile) return "No profile provided.";

  if (typeof profile === "string") return profile;

  const selectedReferees = [1, 2, 3]
    .map(number => {
      if (!profile[`includeReferee${number}`]) return "";

      const name = profile[`referee${number}Name`] || "";
      const email = profile[`referee${number}Email`] || "";
      const phone = profile[`referee${number}Phone`] || "";
      const department = profile[`referee${number}Department`] || "";
      const hospital = profile[`referee${number}Hospital`] || "";

      if (!name && !email && !phone && !department && !hospital) return "";

      return [
        `Referee ${number}`,
        `Name: ${name || "[Add referee name]"}`,
        `Email: ${email || "[Add email]"}`,
        `Phone: ${phone || "[Add phone]"}`,
        `Department / Position: ${department || "[Add department / position]"}`,
        `Hospital / Health Service: ${hospital || "[Add hospital]"}`
      ].join("\n");
    })
    .filter(Boolean)
    .join("\n\n");

  return `
Visa: ${profile.visa || ""}
Needs Sponsorship: ${profile.needsSponsorship ? "Yes" : "No"}
AHPRA: ${profile.ahpra || ""}
Level: ${profile.jobLevel || profile.level || ""}
Preferred State: ${profile.stateFilter || "All states"}
Specialty Interest: ${profile.specialtyInterest || ""}
Willing to Relocate: ${profile.willingToRelocate ? "Yes" : "No"}

Name: ${profile.name || ""}
Email: ${profile.email || ""}
Phone: ${profile.phone || ""}
Location: ${profile.location || ""}
LinkedIn: ${profile.linkedin || ""}

CV Mode: ${profile.cvMode || "Use Structured Template"}

Existing CV:
${profile.baseCv || ""}

Structured CV Template:

Education:
${profile.education || ""}

Professional Summary:
${profile.cvSummary || ""}

Experience:
${profile.experience || profile.cvWorkHistory || ""}

Skills:
${profile.skills || profile.cvClinicalSkills || ""}

Courses / Certifications:
${profile.cvCourses || ""}

Audits / Research / Publications:
${profile.cvAudits || ""}

Extra Notes:
${profile.extraNotes || ""}

Referees:
${profile.cvReferees || ""}

Selected Referees for this application:
${profile.refereesOnRequest ? "Referees available on request" : (selectedReferees || "No referees selected or supplied.")}
`;
}

function buildSearchQueryFromProfile(profile) {
  if (!profile || typeof profile === "string") {
    return "resident medical officer RMO jobs Australia hospital doctor";
  }

  const level = String(profile.jobLevel || profile.level || "RMO").toLowerCase();
  const state = String(profile.stateFilter || "All states");
  const specialty = String(profile.specialtyInterest || "").trim();

  let base = "";

  if (level.includes("intern")) {
    base = "medical intern jobs Australia hospital doctor intern";
  } else if (level.includes("pho")) {
    base = "principal house officer PHO registrar jobs Australia hospital doctor unaccredited registrar";
  } else if (level.includes("registrar")) {
    base = "registrar unaccredited registrar medical officer jobs Australia hospital doctor";
  } else {
    base = "resident medical officer RMO hospital medical officer HMO junior medical officer JMO jobs Australia hospital doctor";
  }

  const location = state && state !== "All states" ? state : "";
  const sponsorship = profile.needsSponsorship
    ? "visa sponsorship sponsor 482 IMG international medical graduate limited registration supervision area of need"
    : "";

  return [base, location, sponsorship, specialty]
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

function buildMedicalContext(profile, job) {
  return `
Doctor Profile:
${profileToText(profile)}

Job Description:
${job}
`;
}


function cvTailoringInstructions() {
  return `
CV tailoring instructions:
- First detect CV Mode from the Doctor Profile.
- If CV Mode is "Use Pasted CV", treat Existing CV as the source of truth. Preserve the candidate's real chronology, roles, qualifications, and wording where appropriate, but improve clarity, structure, and job relevance.
- If CV Mode is "Use Structured Template", build the CV from the structured template fields.
- Do not invent hospitals, dates, qualifications, registration status, visa status, procedures, audits, publications, referees, or achievements.
- If information is missing, include a clear placeholder such as [Add dates] or [Add hospital name] rather than making it up.
- Tailor the CV to the job description by prioritising matching clinical experience, registration/supervision fit, visa/sponsorship fit, ED/hospital/rotational experience, courses, audits, and skills.
- Reorder bullet points within sections so the most relevant material for this job appears first.
- Keep the tone polished, natural, and suitable for Australian hospital medical recruitment.
- Avoid generic AI phrases such as passionate, dynamic, fast-paced, cutting-edge, robust, leverage, and seamless.
- If unsure, prefer omission over guessing.
- Keep formatting clean with clear headings and concise bullet points.
`;
}

function jobCriteriaExtractionInstructions() {
  return `
Before writing the CV, extract the job's key selection criteria internally:
- Role level and specialty area
- Must-have clinical skills
- Preferred clinical skills
- Registration/AHPRA requirements
- Visa/sponsorship clues
- Location/practical fit
- Keywords likely used by recruiters or ATS systems

Then use those criteria to decide what to emphasise in the CV. Do not print this analysis unless specifically asked.
`;
}

function cvQualityInstructions() {
  return `
CV quality rules:
- Write strong, action-based clinical bullet points.
- Convert vague bullets into practical clinical statements while staying truthful.
- Match candidate experience directly to the job requirements.
- Remove or reduce weak, irrelevant, or repetitive content.
- Keep the most relevant experience near the top of each section.
- Do not contradict any information provided in the candidate profile or pasted CV.
- Use a targeted 4 to 5 line professional profile at the top.
- Make the CV sound like it was written by a careful human Australian medical applicant, not an AI tool.
- Prefer direct clinical statements over generic personality claims.
- Preserve the candidate's genuine experience and improve wording without changing the meaning.
`;
}

function humanWritingInstructions() {
  return `
Human writing style rules:
- Write like a careful, real medical applicant, not like an AI generator.
- Use simple, confident, professional language.
- Avoid inflated phrases such as passionate, dynamic, fast-paced, cutting-edge, robust, seamless, leverage, proven track record, highly motivated, and exceptional.
- Avoid over-selling. The tone should be credible, grounded, and clinically mature.
- Use specific clinical duties and real experience from the supplied CV rather than generic claims.
- Keep sentences varied. Do not make every bullet sound the same.
- Prefer practical hospital language: assessed, escalated, documented, coordinated, reviewed, assisted, performed, followed up, communicated, and contributed.
- Do not invent facts, Australian experience, AHPRA status, visa status, hospitals, referees, courses, audits, or procedures.
- If a detail is missing, use a simple placeholder rather than guessing.
`;
}

function topTierMedicalCvInstructions() {
  return `
Create a top-tier Australian hospital medical CV matched to the selected role level.

Role targeting:
- If Level is Intern: emphasise internship readiness, rotations, safe escalation, teamwork, documentation, basic procedures, and willingness to learn.
- If Level is RMO: emphasise independent ward work, ED/acute care exposure, procedural skills, admissions, discharge planning, escalation, multidisciplinary care, and reliability.
- If Level is PHO: emphasise registrar-readiness, leadership, supervision of junior staff, after-hours responsibility, procedural competence, acute decision-making, audits, teaching, and service contribution.

CV format:
- Use a professional Australian hospital CV structure.
- Start with name and contact placeholder if missing.
- Then Professional Profile.
- Then Registration / Visa / Work Rights.
- Then Key Clinical Skills.
- Then Employment History / Clinical Experience.
- Then Education.
- Then Courses / Certifications.
- Then Audits / Research / Publications.
- Then Teaching / Leadership if supplied or relevant.
- Then Referees.

Writing rules:
- Use bullet points beginning with "•" for skills, experience, audits, and achievements.
- Each bullet must start with a strong action verb where possible, such as Managed, Assessed, Coordinated, Performed, Assisted, Documented, Escalated, Led, Supported, or Contributed.
- Keep bullets concise, practical, and clinically relevant.
- Avoid long paragraphs in clinical experience sections.
- Avoid generic AI wording.
- Do not invent anything.
- Use placeholders such as [Add date], [Add hospital], [Add contact details], or [Add referee details] if needed.
- Tailor strongly to the job description and selected role level.
`;
}

function extractClosingDateFromText(text = "") {
  const source = String(text || "");
  const match = source.match(/(?:closing date|applications close|application close date|closes|closing)[:\s-]*([^\n|]+)/i);
  return match?.[1]?.trim() || "";
}

function getClosingStatus(closingDateText = "") {
  if (!closingDateText) return "No closing date found";

  const parsed = Date.parse(closingDateText);
  if (Number.isNaN(parsed)) return "Check closing date";

  const diffDays = Math.ceil((new Date(parsed) - new Date()) / (1000 * 60 * 60 * 24));
  if (diffDays < 0) return "Closed";
  if (diffDays <= 7) return "Closing soon";
  return "Open";
}

function getDaysUntilClosing(closingDateText = "") {
  if (!closingDateText) return null;

  const parsed = Date.parse(closingDateText);
  if (Number.isNaN(parsed)) return null;

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const closingDate = new Date(parsed);
  closingDate.setHours(0, 0, 0, 0);

  return Math.ceil((closingDate - today) / (1000 * 60 * 60 * 24));
}

function addClosingDateInfo(job) {
  const text = `${job.title || ""}\n${job.snippet || ""}\n${job.fullDescription || ""}`;
  const closingDate = extractClosingDateFromText(text);
  const finalClosingDate = job.closingDate || closingDate;
  const daysUntilClosing = getDaysUntilClosing(finalClosingDate);

  return {
    ...job,
    closingDate: finalClosingDate,
    closingStatus: job.closingStatus || getClosingStatus(finalClosingDate),
    daysUntilClosing,
    expiryLabel: daysUntilClosing === null
      ? "Closing date not found"
      : daysUntilClosing < 0
        ? "Expired"
        : daysUntilClosing === 0
          ? "Closes today"
          : daysUntilClosing === 1
            ? "1 day left"
            : `${daysUntilClosing} days left`
  };
}

function scoringFramework() {
  return `
Score each job out of 5 using this framework:

1. Registration fit /1
2. Visa fit /1
3. Level fit /1
4. Training value /1
5. Practical fit /1

Rules:
- If visa is Citizen or Have working rights, do not penalise lack of sponsorship.
- If visa is Requires sponsorship, penalise if sponsorship is not mentioned.
- If AHPRA is General registration, do not penalise basic AHPRA requirements.
- If AHPRA is limited/provisional/eligible only, check supervision and suitability.
- If job is clearly too senior, such as consultant/staff specialist/director, score low for RMO/PHO users.
- If job is nursing/allied health, score 0 and recommend Skip.
- Use only the supplied job data unless the full job description is provided elsewhere in the request.

Apply readiness must be one of:
Ready to apply now / Needs visa clarity / Needs AHPRA/supervision clarity / Not suitable

Return valid JSON array only. Do not include markdown. Do not include explanation outside JSON.
[
  {
    "jobNumber": 1,
    "score": 4.2,
    "recommendation": "Apply",
    "confidence": "Medium",
    "applyReadiness": "Needs visa clarity",
    "breakdown": {
      "registration": 1,
      "visa": 0.5,
      "level": 1,
      "trainingValue": 0.8,
      "practicalFit": 0.9
    },
    "reason": "Short clear reason based only on supplied information.",
    "warning": "Short warning or empty string."
  }
]
`;
}

function makeSeekSlug(query) {
  return query
    .toLowerCase()
    .replace(/\bdoctor\b/g, "")
    .replace(/\bjobs?\b/g, "")
    .replace(/\baustralia\b/g, "")
    .replace(/\bhospital\b/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

async function extractSeekJobsWithPlaywright(url) {
  let browser;

  try {
    browser = await chromium.launch({ headless: true });

    const page = await browser.newPage({
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    });

    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForTimeout(2500);

    const jobs = await page.evaluate(() => {
      const anchors = Array.from(document.querySelectorAll("a[href*='/job/']"));
      const seen = new Set();
      const results = [];

      for (const a of anchors) {
        const href = a.href;
        const title = (a.innerText || "").trim();

        if (!href || !title || title.length < 4) continue;
        if (seen.has(href)) continue;

        seen.add(href);

        let text = "";
        let parent = a.parentElement;

        for (let i = 0; i < 6 && parent; i++) {
          const parentText = (parent.innerText || "").trim();
          if (parentText.length > text.length) text = parentText;
          parent = parent.parentElement;
        }

        const lines = text.split("\n").map(x => x.trim()).filter(Boolean);

        results.push({
          title,
          employer: "",
          location: "",
          jobType: "",
          link: href,
          snippet: lines.slice(0, 18).join(" | ")
        });
      }

      return results;
    });

    return jobs.slice(0, 100);
  } catch (error) {
    console.error("Playwright SEEK extraction failed:", error.message);
    return [];
  } finally {
    if (browser) await browser.close();
  }
}

function isSeekSearchPage(url) {
  const lower = url.toLowerCase();
  return lower.includes("seek.com.au") && !lower.includes("seek.com.au/job/");
}

function isAllowedUrl(url) {
  const allowed = [
    "seek.com.au",
    "smartjobs.qld.gov.au",
    "apply-springboard.health.qld.gov.au",
    "jobs.health.nsw.gov.au",
    "au.jora.com",
    "jora.com",
    "healthworkforce.com.au",
    "medrecruit.com",
    "skilledmedical.com",
    "headmedical.com",
    "globalmedics.com.au",
    "ramsaycareers.com.au",
    "healthscope.com.au"
  ];

  const blocked = [
    "facebook.com",
    "reddit.com",
    "ahpra.gov.au",
    "medicalboard.gov.au",
    "youtube.com",
    "linkedin.com"
  ];

  const lower = url.toLowerCase();

  return allowed.some(d => lower.includes(d)) && !blocked.some(d => lower.includes(d));
}

function getHardRejectionReason(job) {
  const title = `${job.title || ""}`.toLowerCase();
  const text = `${job.title || ""} ${job.snippet || ""} ${job.employer || ""}`.toLowerCase();

  const blockedTitleTerms = [
    "nurse",
    "nursing",
    "registered nurse",
    "clinical nurse",
    "enrolled nurse",
    "midwife",
    "midwifery",
    "assistant in nursing",
    "support worker",
    "personal care worker",
    "pharmacist",
    "physiotherapist",
    "occupational therapist",
    "psychologist",
    "social worker",
    "dentist",
    "veterinarian",
    "sonographer",
    "radiographer",
    "speech pathologist",
    "dietitian",
    "dental assistant"
  ];

  const tooSeniorForJuniorStream = [
    "staff specialist",
    "consultant physician",
    "consultant psychiatrist",
    "consultant anaesthetist",
    "consultant anesthetist",
    "clinical director",
    "medical director",
    "director of medical services",
    "executive director",
    "specialist consultant"
  ];

  if (blockedTitleTerms.some(term => title.includes(term))) {
    return "Skipped: non-doctor or allied health role.";
  }

  if (tooSeniorForJuniorStream.some(term => text.includes(term))) {
    return "Skipped: consultant/staff specialist/director-level role, not suitable for RMO/PHO stream.";
  }

  if (text.includes("volunteer") || text.includes("unpaid")) {
    return "Skipped: unpaid or volunteer role.";
  }

  if (text.includes("telehealth only") || text.includes("remote only")) {
    return "Skipped: remote/telehealth-only role, likely not suitable for supervised hospital medical work.";
  }

  return "";
}

function isClearlyNonDoctorJob(job) {
  return Boolean(getHardRejectionReason(job));
}

function hasDoctorSignal(job) {
  const text = `${job.title || ""} ${job.snippet || ""}`.toLowerCase();

  const terms = [
    "rmo",
    "resident medical officer",
    "hmo",
    "hospital medical officer",
    "jmo",
    "junior medical officer",
    "pho",
    "principal house officer",
    "medical officer",
    "doctor",
    "registrar",
    "intern",
    "unaccredited registrar",
    "medical practitioner"
  ];

  return terms.some(term => text.includes(term));
}

function getInstantScore(job, userQuery = "") {
  const title = `${job.title || ""}`.toLowerCase();
  const text = `${job.title || ""} ${job.snippet || ""} ${job.location || ""}`.toLowerCase();
  const query = String(userQuery || "").toLowerCase();

  let score = 0;

  if (title.includes("resident medical officer") || title.includes("rmo")) score += 35;
  if (title.includes("hospital medical officer") || title.includes("hmo")) score += 30;
  if (title.includes("junior medical officer") || title.includes("jmo")) score += 28;
  if (title.includes("principal house officer") || title.includes("pho")) score += 32;
  if (title.includes("unaccredited registrar")) score += 24;
  if (title.includes("registrar")) score += 18;
  if (title.includes("medical officer")) score += 22;
  if (title.includes("intern")) score += 18;

  if (text.includes("hospital")) score += 8;
  if (text.includes("health service")) score += 6;
  if (text.includes("queensland health") || text.includes("nsw health") || text.includes("sa health") || text.includes("wa health")) score += 8;

  if (text.includes("emergency") || text.includes("ed ") || text.includes("emergency department")) score += 5;
  if (text.includes("medicine") || text.includes("surgery") || text.includes("icu") || text.includes("critical care")) score += 4;
  if (text.includes("rotation") || text.includes("rotational")) score += 5;
  if (text.includes("supervision") || text.includes("supervised")) score += 8;
  if (text.includes("limited registration") || text.includes("img") || text.includes("international medical graduate")) score += 10;
  if (text.includes("sponsor") || text.includes("sponsorship") || text.includes("482") || text.includes("visa")) score += 10;

  if (query.includes("queensland") && (text.includes("queensland") || text.includes("qld"))) score += 10;
  if (query.includes("new south wales") && (text.includes("new south wales") || text.includes("nsw"))) score += 10;
  if (query.includes("victoria") && (text.includes("victoria") || text.includes("vic"))) score += 10;
  if (query.includes("south australia") && (text.includes("south australia") || text.includes("sa health"))) score += 10;
  if (query.includes("western australia") && (text.includes("western australia") || text.includes("wa health"))) score += 10;

  if (text.includes("casual")) score -= 8;
  if (text.includes("telehealth")) score -= 12;
  if (text.includes("general practitioner") || text.includes("gp only")) score -= 10;
  if (text.includes("senior medical officer") || title.includes("smo")) score -= 12;
  if (text.includes("staff specialist") || text.includes("consultant")) score -= 25;

  return Math.max(0, score);
}

function getInstantReadiness(job) {
  const text = `${job.title || ""} ${job.snippet || ""}`.toLowerCase();
  const hardRejectionReason = getHardRejectionReason(job);

  if (hardRejectionReason) {
    return "Not suitable";
  }

  if (
    text.includes("limited registration") ||
    text.includes("supervision") ||
    text.includes("img") ||
    text.includes("international medical graduate") ||
    text.includes("sponsor") ||
    text.includes("sponsorship") ||
    text.includes("482") ||
    text.includes("visa")
  ) {
    return "High potential";
  }

  if (
    text.includes("rmo") ||
    text.includes("resident medical officer") ||
    text.includes("hmo") ||
    text.includes("hospital medical officer") ||
    text.includes("pho") ||
    text.includes("registrar")
  ) {
    return "Worth reviewing";
  }

  return "Needs manual check";
}

function enrichAndRankJobs(jobs, userQuery = "") {
  return jobs
    .map(job => {
      const instantScore = getInstantScore(job, userQuery);
      const instantReadiness = getInstantReadiness(job);
      const hardRejectionReason = getHardRejectionReason(job);

      return addClosingDateInfo({
        ...job,
        instantScore,
        instantReadiness,
        hardRejectionReason
      });
    })
    .sort((a, b) => b.instantScore - a.instantScore);
}

function normaliseJobKey(job) {
  const title = String(job.title || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const employer = String(job.employer || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const location = String(job.location || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const linkHost = (() => {
    try {
      return new URL(job.link).hostname.replace(/^www\./, "");
    } catch {
      return "";
    }
  })();

  return `${title}|${employer}|${location}|${linkHost}`;
}

function splitAndDedupeJobs(jobs, userQuery = "") {
  const seen = new Set();
  const unique = [];
  const rejected = [];

  for (const job of jobs) {
    if (!job.title || !job.link) continue;

    const hardRejectionReason = getHardRejectionReason(job);
    if (hardRejectionReason) {
      rejected.push({
        ...job,
        numericScore: 0,
        instantScore: 0,
        instantReadiness: "Not suitable",
        hardRejectionReason,
        aiScore: hardRejectionReason
      });
      continue;
    }

    if (!hasDoctorSignal(job)) {
      rejected.push({
        ...job,
        numericScore: 0,
        instantScore: 0,
        instantReadiness: "Not suitable",
        hardRejectionReason: "Skipped: no clear doctor/RMO/HMO/PHO/registrar signal in title or snippet.",
        aiScore: "Skipped: no clear doctor/RMO/HMO/PHO/registrar signal in title or snippet."
      });
      continue;
    }

    const key = normaliseJobKey(job);
    if (seen.has(key)) continue;

    seen.add(key);
    unique.push(job);
  }

  return {
    suitable: enrichAndRankJobs(unique, userQuery),
    rejected: enrichAndRankJobs(rejected, userQuery)
  };
}

function dedupeJobs(jobs, userQuery = "") {
  return splitAndDedupeJobs(jobs, userQuery).suitable;
}

function createDocxFromText(text, filename, res) {
  const lines = text.split("\n");
  const children = [];

  const headingWords = [
    "personal details",
    "professional profile",
    "registration",
    "visa",
    "work rights",
    "key clinical skills",
    "clinical skills",
    "employment history",
    "clinical experience",
    "education",
    "qualifications",
    "courses",
    "certifications",
    "audit",
    "audits",
    "research",
    "publications",
    "teaching",
    "leadership",
    "achievements",
    "referees",
    "cover letter",
    "application email"
  ];

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      children.push(new Paragraph({ text: "", spacing: { after: 120 } }));
      continue;
    }

    const lower = trimmed.toLowerCase();

    const isHeading =
      headingWords.some(h => lower.includes(h)) &&
      trimmed.length < 90 &&
      !trimmed.startsWith("•") &&
      !trimmed.startsWith("-");

    if (isHeading) {
      children.push(
        new Paragraph({
          children: [
            new TextRun({
              text: trimmed.toUpperCase(),
              bold: true,
              size: 26
            })
          ],
          heading: HeadingLevel.HEADING_2,
          spacing: { before: 300, after: 120 }
        })
      );
      continue;
    }

    if (trimmed.startsWith("•") || trimmed.startsWith("-")) {
      children.push(
        new Paragraph({
          children: [
            new TextRun({
              text: trimmed.replace(/^[-•]\s*/, ""),
              size: 22
            })
          ],
          bullet: { level: 0 },
          spacing: { after: 80 }
        })
      );
      continue;
    }

    const looksLikeName =
      children.length === 0 ||
      (
        children.length < 3 &&
        trimmed.length < 60 &&
        !trimmed.includes(".") &&
        !trimmed.includes(":") &&
        /^[A-Z][A-Za-z\s.'-]+$/.test(trimmed)
      );

    if (looksLikeName && children.length < 3) {
      children.push(
        new Paragraph({
          children: [
            new TextRun({
              text: trimmed,
              bold: true,
              size: 32
            })
          ],
          alignment: AlignmentType.CENTER,
          spacing: { after: 160 }
        })
      );
      continue;
    }

    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text: trimmed,
            size: 22
          })
        ],
        spacing: { after: 100 }
      })
    );
  }

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            margin: {
              top: 720,
              right: 720,
              bottom: 720,
              left: 720
            }
          }
        },
        children
      }
    ]
  });

  return Packer.toBuffer(doc).then(buffer => {
    res.setHeader("Content-Disposition", `attachment; filename=${filename}`);
    res.setHeader("Content-Type", "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
    res.send(buffer);
  });
}

app.post("/auto-search", async (req, res) => {
  try {
    const { query, profile } = req.body;

    if (!TAVILY_API_KEY) {
      throw new Error("Missing TAVILY_API_KEY in .env file.");
    }

    const userQuery = query || buildSearchQueryFromProfile(profile);
    console.log("AUTO SEARCH START");
    console.log("Query used:", userQuery);

    const cacheKey = `search:${userQuery.toLowerCase()}`;

    const cached = searchCache.get(cacheKey);
    if (cached) {
      console.log("Returning cached jobs:", cached.length);
      return res.json({ results: cached, cached: true, queryUsed: userQuery });
    }

    const searchQuery = `${userQuery} -nurse -nursing -midwife -pharmacist -physiotherapist SEEK SmartJobs NSW Health medical jobs`;

    const tavilyResponse = await tavilyClient.search(searchQuery, {
      searchDepth: "basic",
      maxResults: 50,
      includeAnswer: false,
      includeRawContent: false,
      topic: "general"
    });
    console.log("Tavily raw results:", tavilyResponse.results?.length || 0);

    let extractedJobs = [];

    const slug = makeSeekSlug(userQuery) || "rmo";
    const seekBaseUrl = `https://www.seek.com.au/${slug}-jobs`;

    const seekUrls = [
      seekBaseUrl,
      `${seekBaseUrl}?page=2`,
      `${seekBaseUrl}?page=3`
    ];

    const seekJobGroups = await Promise.all(
      seekUrls.map(url => extractSeekJobsWithPlaywright(url))
    );

    for (const seekJobs of seekJobGroups) {
      extractedJobs.push(...seekJobs);
    }
    console.log("Playwright SEEK direct jobs:", extractedJobs.length);

    const tavilyResults = (tavilyResponse.results || [])
      .filter(item => isAllowedUrl(item.url || ""))
      .slice(0, 40);

    const seekPages = tavilyResults
      .filter(item => isSeekSearchPage(item.url || ""))
      .slice(0, 2);

    const tavilySeekJobGroups = await Promise.all(
      seekPages.map(page => extractSeekJobsWithPlaywright(page.url))
    );

    for (const seekJobs of tavilySeekJobGroups) {
      extractedJobs.push(...seekJobs);
    }
    console.log("After Tavily SEEK page scraping jobs:", extractedJobs.length);

    const directJobs = tavilyResults
      .filter(item => !isSeekSearchPage(item.url || ""))
      .map(item => ({
        title: item.title || "Untitled job",
        employer: "",
        location: "",
        jobType: "",
        link: item.url,
        snippet: item.content || ""
      }));

    extractedJobs.push(...directJobs);
    console.log("Extracted jobs before dedupe:", extractedJobs.length);

    const uniqueJobs = dedupeJobs(extractedJobs, userQuery).slice(0, 200);
    console.log("Unique jobs after dedupe:", uniqueJobs.length);

    searchCache.set(cacheKey, uniqueJobs);

    res.json({ results: uniqueJobs, queryUsed: userQuery });

  } catch (error) {
    res.status(500).json({
      error: `Auto-search error: ${error.message}`
    });
  }
});

app.post("/score-jobs", async (req, res) => {
  try {
    const { jobs, quickProfile, profile } = req.body;
    const applicantProfile = quickProfile || profileToText(profile);

    if (!Array.isArray(jobs) || jobs.length === 0) {
      return res.status(400).json({ error: "No jobs received for scoring." });
    }

    const splitJobs = splitAndDedupeJobs(jobs, applicantProfile);
    const jobsToScore = await enrichTopJobsWithFirecrawl(splitJobs.suitable.slice(0, 20));
    const cacheKey = `score:${SCORING_PROVIDER}:${JSON.stringify(jobsToScore.map(j => j.link))}:${applicantProfile}`;

    const cached = scoreCache.get(cacheKey);
    if (cached) return res.json({ results: cached.results || cached, rejected: cached.rejected || [], cached: true });

    const prompt = `
You are an Australian medical recruitment assistant.

Give PRELIMINARY suitability scoring for each job.
Use only the supplied title, link, employer, location, type, and snippet.
Do not pretend you have read the full official job description.

Applicant:
${applicantProfile || "No profile provided"}

${scoringFramework()}

Jobs:
${jobsToScore.map((job, index) => `
Job ${index + 1}
Title: ${job.title}
Employer: ${job.employer || ""}
Location: ${job.location || ""}
Type: ${job.jobType || ""}
Link: ${job.link}
Snippet: ${job.snippet || ""}
Full job description source: ${job.descriptionSource || "snippet"}
Full job description if available: ${(job.fullDescription || "").slice(0, 6000)}
Closing date: ${job.closingDate || "Not stated"}
Closing status: ${job.closingStatus || "No closing date found"}
`).join("\n")}
`;

    const text = await askScoringModel(prompt);
    const scores = extractJson(text);

    const scoredJobs = jobsToScore
      .map((job, index) => {
        const score = scores.find(s => Number(s.jobNumber) === index + 1);
        const aiScore = buildScoreText(score);
        const numericScore = Number(score?.score || parseScore(aiScore) || 0);

        return addClosingDateInfo({
          ...job,
          aiScore,
          numericScore,
          scoringProvider: SCORING_PROVIDER,
          descriptionSource: job.descriptionSource || "snippet",
          instantReadiness: job.instantReadiness || getInstantReadiness(job),
          instantScore: job.instantScore || getInstantScore(job),
          hardRejectionReason: job.hardRejectionReason || ""
        });
      })
      .sort((a, b) => b.numericScore - a.numericScore);

    scoreCache.set(cacheKey, { results: scoredJobs, rejected: splitJobs.rejected });

    res.json({ results: scoredJobs, rejected: splitJobs.rejected });
  } catch (error) {
    res.status(500).json({ error: `Score error: ${error.message}` });
  }
});

app.post("/application-pack", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
You are an Australian medical recruitment assistant.

Prepare a semi-automated application pack for this medical job.
Use a natural, human Australian medical recruitment tone. Avoid robotic phrasing and generic filler.

Do NOT pretend the application has been submitted.
Do NOT answer legal/visa/AHPRA declaration questions automatically.
The applicant must review everything before submitting.

Return:

1. APPLICATION READINESS CHECK
- Apply / Maybe / Skip
- Key reasons
- Registration concerns
- Visa concerns

2. TAILORED CV STRATEGY
- State whether the profile is using pasted CV mode or structured template mode
- Extract the key job criteria in simple bullet points
- What to emphasise for this exact job
- What to de-emphasise or remove
- Which sections should be moved higher
- Missing information to add
- Suggested targeted professional profile paragraph
- Suggested 6 to 10 high-impact CV bullet points matched to the job

3. COVER LETTER
Write a complete professional Australian medical cover letter.

4. SHORT APPLICATION EMAIL
Write a short email to Medical Workforce / recruitment team.

5. SELECTION CRITERIA / STATEMENT RESPONSES
Write short responses using Australian hospital language.

6. QUESTIONS TO ASK HR
Include questions about sponsorship, AHPRA supervision, start date, contract, and referees.

7. FINAL CHECKLIST
Give a practical checklist before applying.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("application-pack", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: `Application pack error: ${error.message}` });
  }
});

app.post("/application-pack-download", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
Prepare a complete semi-automated Australian medical job application pack.
Use a polished, natural, human Australian medical recruitment tone. Avoid robotic phrasing and generic filler.

Return plain text only.
Include:
- Application readiness check
- Tailored CV strategy, including whether pasted CV mode or structured template mode was used
- Cover letter
- Short application email
- Selection criteria responses
- Questions to ask HR
- Final checklist

Do not say the application has been submitted.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("application-pack-download", profile, job, MODEL_SMART);
    const text = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    await createDocxFromText(text, "Application_Pack.docx", res);
  } catch (error) {
    res.status(500).json({ error: `Application pack download error: ${error.message}` });
  }
});

app.post("/upload-docx", async (req, res) => {
  try {
    if (!req.body.file) return res.status(400).json({ error: "No Word file received." });

    const buffer = Buffer.from(req.body.file, "base64");
    const result = await mammoth.extractRawText({ buffer });
    res.json({ text: result.value });
  } catch (error) {
    res.status(500).json({ error: `Word upload error: ${error.message}` });
  }
});

app.post("/evaluate", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
You are an expert Australian medical recruiter.

This is a FINAL evaluation because the full job description has been provided.
Use a concise, practical Australian medical recruitment style.

${scoringFramework()}

Also include:
- Registration fit explanation
- Visa/sponsorship fit explanation
- Key risks/red flags
- Final recommendation

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("evaluate", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: `Evaluate error: ${error.message}` });
  }
});

app.post("/cv", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
You are an expert Australian hospital medical CV writer.

${jobCriteriaExtractionInstructions()}
${cvTailoringInstructions()}
${cvQualityInstructions()}
${humanWritingInstructions()}
${topTierMedicalCvInstructions()}

Output requirements:
- Return the full tailored CV only.
- Use Australian medical CV headings.
- Use bullet points beginning with "•" for clinical experience, skills, audits, achievements, teaching, and leadership.
- Include a targeted professional profile at the top.
- Include key clinical skills relevant to the job and selected role level.
- Include employment history in reverse chronological order if dates are supplied.
- Include education, registration/visa information if supplied, courses, audits/research/publications, and referees if supplied.
- Keep it honest, polished, and professional.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cv", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: `CV error: ${error.message}` });
  }
});

app.post("/cv-download", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
You are an expert Australian hospital medical CV writer.

Create a top-tier Australian medical CV suitable for Word download using the candidate's selected CV mode.

${jobCriteriaExtractionInstructions()}
${cvTailoringInstructions()}
${cvQualityInstructions()}
${humanWritingInstructions()}
${topTierMedicalCvInstructions()}

Rules:
- Return plain text only.
- Do not use markdown symbols.
- Use clear section headings.
- Use bullet points starting with "•" for clinical experience, skills, audits, achievements, teaching, and leadership.
- Return the full CV only.
- Do not fake registration or visa status.
- Tailor strongly to the supplied job description and selected role level.
- Use placeholders for missing facts rather than inventing details.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cv-download", profile, job, MODEL_SMART);
    const text = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    const profileText = profileToText(profile);
    const nameMatch = String(profileText || "").match(/Name:\s*(.*)/);
    const levelMatch = String(profileText || "").match(/Level:\s*(.*)/);
    const safeName = (nameMatch?.[1] || "Medical")
      .trim()
      .replace(/[^a-z0-9]+/gi, "_")
      .replace(/^_+|_+$/g, "") || "Medical";
    const safeLevel = (levelMatch?.[1] || "CV")
      .trim()
      .replace(/[^a-z0-9]+/gi, "_")
      .replace(/^_+|_+$/g, "") || "CV";

    await createDocxFromText(text, `${safeName}_${safeLevel}_CV.docx`, res);
  } catch (error) {
    res.status(500).json({ error: `CV download error: ${error.message}` });
  }
});

app.post("/cv-review", async (req, res) => {
  try {
    const { profile, job, cvText } = req.body;

    const prompt = `
You are an Australian medical recruitment reviewer.

Review this CV against the supplied job description.

Return:
1. CV match score out of 5
2. Key strengths
3. Main gaps
4. Missing keywords or criteria
5. Suggested changes to improve match
6. 5 stronger bullet points the applicant could use, without inventing facts

Rules:
- Be practical and concise.
- Do not invent experience.
- If facts are missing, say what to add as placeholders.
- Focus on Australian hospital medical recruitment.

Doctor Profile:
${profile || ""}

Job Description:
${job || ""}

CV to review:
${cvText || ""}
`;

    const cacheKey = makeGeminiCacheKey("cv-review", profile, `${job || ""}:${cvText || ""}`, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: `CV review error: ${error.message}` });
  }
});

app.post("/job-criteria", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
You are an Australian medical recruitment assistant.

Extract key criteria from this Australian medical job.

Return:
1. Role level and specialty
2. Must-have criteria
3. Preferred criteria
4. AHPRA/registration clues
5. Visa/sponsorship clues
6. Location and practical considerations
7. Closing date if mentioned
8. Keywords for CV/cover letter
9. Candidate fit notes based on the supplied profile

Rules:
- Be concise and practical.
- Do not invent requirements.
- If unsure, say "Not stated".

${buildMedicalContext(profile, job)}
`;

    const result = await askGeminiCached(
      makeGeminiCacheKey("job-criteria", profile, job, MODEL_SMART),
      prompt,
      MODEL_SMART
    );

    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: `Job criteria error: ${error.message}` });
  }
});

app.post("/cv-improve", async (req, res) => {
  try {
    const { profile, job, cvText, reviewText } = req.body;

    const prompt = `
You are an expert Australian medical CV editor.

Improve this CV for the supplied Australian medical job.

Rules:
- Return the full improved CV only.
- Do not use markdown symbols.
- Do not invent facts, dates, hospitals, qualifications, registration status, visa status, audits, publications, referees, or achievements.
- Use placeholders for missing details, such as [Add date], [Add hospital], or [Add referee details].
- Improve the professional summary, clinical bullet points, section order, and keyword match.
- Use the CV review feedback if supplied.
- If unsure, prefer omission over guessing.
- Keep the tone natural and suitable for Australian hospital medical recruitment.

Doctor Profile:
${profile || ""}

Job Description:
${job || ""}

CV Review Feedback:
${reviewText || ""}

CV to improve:
${cvText || ""}
`;

    const result = await askGeminiCached(
      makeGeminiCacheKey("cv-improve", profile, `${job || ""}:${cvText || ""}:${reviewText || ""}`, MODEL_SMART),
      prompt,
      MODEL_SMART
    );

    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: `CV improve error: ${error.message}` });
  }
});

app.post("/cover-letter", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
Write an Australian medical cover letter.
First identify the job's key criteria internally, then match the applicant's real experience to those criteria.
Use a polished, natural, human tone. Avoid generic AI-sounding statements.
${humanWritingInstructions()}

Cover letter style:
- Sound like a real doctor applying for the job, not a template.
- Keep the opening direct and specific to the role.
- Use the applicant's real rotations, clinical duties, AMC result, English test, registration status, and visa/sponsorship situation where relevant.
- Do not overstate Australian experience if the applicant has not worked in Australia.
- Make sponsorship/AHPRA supervision wording honest and professional.
- Keep it warm, clear, and concise.

Important:
- Do not fake registration or visa status.
- Tailor it to the job description.
- If unsure, prefer omission over guessing.

Return:
1. Cover letter
2. Short email text

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cover-letter", profile, job, MODEL_SMART);
    const result = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    res.json({ result });
  } catch (error) {
    res.status(500).json({ error: `Cover letter error: ${error.message}` });
  }
});

app.post("/cover-letter-download", async (req, res) => {
  try {
    const { profile, job } = req.body;

    const prompt = `
Write an Australian medical cover letter.
First identify the job's key criteria internally, then match the applicant's real experience to those criteria.
Use a polished, natural, human tone. Avoid generic AI-sounding statements.
${humanWritingInstructions()}

Cover letter style:
- Sound like a real doctor applying for the job, not a template.
- Keep the opening direct and specific to the role.
- Use the applicant's real rotations, clinical duties, AMC result, English test, registration status, and visa/sponsorship situation where relevant.
- Do not overstate Australian experience if the applicant has not worked in Australia.
- Make sponsorship/AHPRA supervision wording honest and professional.
- Keep it warm, clear, and concise.

Rules:
- Return plain text only.
- Do not use markdown symbols.
- Do not fake registration or visa status.
- Tailor it to the supplied job description.
- If unsure, prefer omission over guessing.
- Include a short application email after the cover letter.

${buildMedicalContext(profile, job)}
`;

    const cacheKey = makeGeminiCacheKey("cover-letter-download", profile, job, MODEL_SMART);
    const text = await askGeminiCached(cacheKey, prompt, MODEL_SMART);
    const profileText = profileToText(profile);
    const nameMatch = String(profileText || "").match(/Name:\s*(.*)/);
    const levelMatch = String(profileText || "").match(/Level:\s*(.*)/);
    const safeName = (nameMatch?.[1] || "Medical")
      .trim()
      .replace(/[^a-z0-9]+/gi, "_")
      .replace(/^_+|_+$/g, "") || "Medical";
    const safeLevel = (levelMatch?.[1] || "Cover")
      .trim()
      .replace(/[^a-z0-9]+/gi, "_")
      .replace(/^_+|_+$/g, "") || "Cover";

    await createDocxFromText(text, `${safeName}_${safeLevel}_Cover_Letter.docx`, res);
  } catch (error) {
    res.status(500).json({ error: `Cover letter download error: ${error.message}` });
  }
});

app.listen(3000, () => {
  console.log(`Server running with 200-job search. Scoring provider: ${SCORING_PROVIDER}. Firecrawl top jobs: ${FIRECRAWL_TOP_N}. Gemini CV/evaluation model: ${MODEL_SMART}. http://localhost:3000`);
});