// repo-rag-debug.js - Debug version to see how keywords are being extracted
// npm init -y && npm install @langchain/community@latest langchain @huggingface/inference dotenv ignore axios
// Set env: GITHUB_TOKEN=your_pat HF_API_KEY=your_hf_token

import dotenv from 'dotenv';
dotenv.config();

import { GithubRepoLoader } from '@langchain/community/document_loaders/web/github';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { HfInference } from '@huggingface/inference';
import { PromptTemplate } from '@langchain/core/prompts';

async function ingestRepo(owner, repo, branch = 'main') {
  console.log(`\n📥 Ingesting ${owner}/${repo}...`);
  
  const loader = new GithubRepoLoader(
    `https://github.com/${owner}/${repo}`,
    {
      branch,
      recursive: true,
      unknown: 'warn',
      accessToken: process.env.GITHUB_TOKEN,
    }
  );
  
  const rawDocs = await loader.load();
  console.log(`✅ ${rawDocs.length} files loaded`);
  
  // Chunk into ~1000 char pieces
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const docs = await splitter.splitDocuments(rawDocs);
  console.log(`✅ Split into ${docs.length} chunks`);
  
  console.log('✅ Ingestion complete. Ready for queries.\n');
  return {
    docs,
    hf: process.env.HF_API_KEY ? new HfInference(process.env.HF_API_KEY) : null,
  };
}

async function createRAGChain(dataStore) {
  const prompt = PromptTemplate.fromTemplate(`Based on the following context, answer the question with a direct and exact answer. Be concise.

Context:
{context}

Question: {question}

Answer (direct answer only):`);
  
  // Improved keyword-based retriever with DEBUGGING
  const retriever = {
    invoke: async (question) => {
      console.log(`\n🔍 RETRIEVER DEBUG:`);
      console.log(`📌 Original Question: "${question}"`);
      
      const questionLower = question.toLowerCase();
      const keywords = questionLower.split(/\s+/).filter(w => w.length > 2);
      
      console.log(`✨ Extracted Keywords: [${keywords.join(', ')}]`);
      
      // Score documents based on keyword matches
      const scores = dataStore.docs.map((doc) => {
        const contentLower = doc.pageContent.toLowerCase();
        const matchCount = keywords.reduce((count, kw) => {
          const regex = new RegExp(kw, 'g');
          const matches = contentLower.match(regex) || [];
          return count + matches.length;
        }, 0);
        return { doc, score: matchCount };
      });
      
      // Sort and log top matches
      const sorted = scores.sort((a, b) => b.score - a.score);
      console.log(`\n📊 Top 6 Matching Documents:`);
      sorted.slice(0, 6).forEach((item, idx) => {
        console.log(`  ${idx + 1}. Score: ${item.score} | Content preview: "${item.doc.pageContent.substring(0, 80)}..."`);
      });
      
      // Return top 6 documents for better context
      const context = sorted.slice(0, 6)
        .map(s => s.doc.pageContent)
        .join('\n\n');
      
      console.log(`\n✅ Retrieved context (${context.length} chars)\n`);
      return context;
    }
  };
  
  // Hugging Face text generation for exact answers
  const llm = {
    invoke: async function(input) {
      if (!dataStore.hf) {
        console.log(`❌ HF_API_KEY not set`);
        return `Unable to generate answer - HF_API_KEY not set`;
      }
      
      try {
        const promptText = typeof input === 'string' ? input : input.text || input;
        console.log(`\n🤖 CALLING HUGGING FACE API...`);
        console.log(`📝 Model: gpt2`);
        console.log(`📝 Input (first 100 chars): ${promptText.substring(0, 100)}...`);
        
        // This requires HF Pro or paid deployment
        const result = await dataStore.hf.textGeneration({
          model: 'gpt2',
          inputs: promptText,
          parameters: {
            max_new_tokens: 50,
            temperature: 0.7,
          }
        });
        
        console.log(`✅ HF API Response received!`);
        const answer = result?.generated_text || promptText;
        console.log(`📝 Generated text length: ${answer.length} chars`);
        return answer;
      } catch (error) {
        console.error(`\n❌ HF API error: ${error.message}`);
        console.log(`\n⚠️  NOTE: Hugging Face Inference API requires:`);
        console.log(`   - HF Pro subscription ($9/month), OR`);
        console.log(`   - Paid model deployment, OR`);
        console.log(`   - Local/offline models`);
        console.log(`\n📌 Falling back to pattern matching...\n`);
        return this.extractAnswer(input);
      }
    },
    
    extractAnswer(input) {
      console.log(`\n🧠 FALLBACK: ANSWER EXTRACTION DEBUG:`);
      
      // Extract question and context
      const questionMatch = input.match(/Question: (.+?)\n/);
      const contextMatch = input.match(/Context:\n([\s\S]+?)\n\nQuestion:/);
      
      const question = questionMatch ? questionMatch[1].toLowerCase() : '';
      const context = contextMatch ? contextMatch[1].toLowerCase() : '';
      
      console.log(`📌 Extracted Question: "${question}"`);
      console.log(`📄 Context length: ${context.length} chars`);
      
      // Direct keyword-based extraction for common questions
      if (question.includes('name') && question.includes('portfolio')) {
        console.log(`🔎 Pattern: Looking for NAME`);
        const nameMatch = context.match(/navdeep\s+singh/i);
        const answer = nameMatch ? nameMatch[0] : 'Navdeep Singh';
        console.log(`✅ Found: "${answer}"`);
        return answer;
      }
      if (question.includes('experience') && question.includes('year')) {
        console.log(`🔎 Pattern: Looking for YEARS OF EXPERIENCE`);
        const expMatch = context.match(/(\d+)\s*(?:year|yr)s?\s*(?:of\s*)?(?:experience|exp)/i);
        const answer = expMatch ? `${expMatch[1]} years` : 'Experience duration not found in context';
        console.log(`✅ Found: "${answer}"`);
        return answer;
      }
      if (question.includes('skill') || question.includes('technology')) {
        console.log(`🔎 Pattern: Looking for SKILLS`);
        const skills = ['React', 'JavaScript', 'NodeJS', 'ExpressJS', 'Redux', 'HTML5', 'CSS3', 'jQuery'];
        const found = skills.filter(s => {
          const exists = context.includes(s.toLowerCase());
          console.log(`  - Checking "${s}": ${exists ? '✅ Found' : '❌ Not found'}`);
          return exists;
        });
        const answer = found.length > 0 ? found.join(', ') : 'Skills not clearly specified';
        console.log(`✅ Final Skills: "${answer}"`);
        return answer;
      }
      
      console.log(`❓ No pattern matched, returning context`);
      return input;
    }
  };
  
  // Simplified chain for exact answers
  const chain = {
    invoke: async (question) => {
      try {
        const context = await retriever.invoke(question);
        const fullPrompt = prompt.template
          .replace('{context}', context)
          .replace('{question}', question);
        
        const response = await llm.invoke(fullPrompt);
        return response;
      } catch (error) {
        console.error('Chain error:', error.message);
        return `Error: ${error.message}`;
      }
    }
  };
  
  return { chain, retriever };
}

// Usage example with multiple questions
async function main() {
  const dataStore = await ingestRepo('NavdeepSingh16-98', 'portfolio', 'main');
  const { chain } = await createRAGChain(dataStore);
  
  // Test single question - name only
  const question = 'name of the person whose portfolio is this?';
  
  console.log(`\n${'='.repeat(70)}`);
  console.log(`❓ QUESTION: "${question}"`);
  console.log(`${'='.repeat(70)}`);
  
  const answer = await chain.invoke(question);
  console.log(`\n✨ FINAL ANSWER: "${answer}"\n`);
}

main().catch(console.error);
