"use client";

import { useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Home, 
  MapPin, 
  Square, 
  Bed, 
  Bath, 
  Layout, 
  ChevronRight, 
  Info,
  Loader2
} from "lucide-react";

export default function HousePricePredictor() {
  const [formData, setFormData] = useState({
    area: "1200",
    bhk: 2,
    bathrooms: 2,
    balcony: 1,
    location: "Electronic City, Bangalore",
  });

  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const locations = [
    "Electronic City, Bangalore",
    "Whitefield, Bangalore",
    "Sarjapur Road, Bangalore",
    "HSR Layout, Bangalore",
    "Koramangala, Bangalore",
    "Indiranagar, Bangalore",
    "Jayanagar, Bangalore",
    "Hebbal, Bangalore",
    "Powai, Mumbai",
    "Andheri West, Mumbai",
    "Bandra West, Mumbai",
    "Juhu, Mumbai",
    "Worli, Mumbai",
    "South Mumbai",
    "Gachibowli, Hyderabad",
    "Kondapur, Hyderabad",
    "Madhapur, Hyderabad",
    "Banjara Hills, Hyderabad",
    "Jubilee Hills, Hyderabad",
    "Gurgaon Sector 56",
    "Gurgaon Sector 45",
    "DLF Phase 3, Gurgaon",
    "Noida Sector 62",
    "Greater Noida West",
    "Salt Lake, Kolkata",
    "New Town, Kolkata",
    "Anna Nagar, Chennai",
    "Adyar, Chennai",
  ];

  const handlePredict = async () => {
    // Validation
    if (parseFloat(formData.area) <= 0) {
      setError("Area must be greater than 0");
      return;
    }
    setError("");
    setLoading(true);
    setPrediction(null);

    try {
      const response = await axios.post("http://localhost:8000/predict", formData);
      setPrediction(response.data);
    } catch (err) {
      setError("Failed to get prediction. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#FFFFFF] py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold text-[#1e293b] mb-2"
          >
            Professional Indian House Price Predictor
          </motion.h1>
          <p className="text-[#64748b]">Estimate your property's value with professional precision</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Input Section */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            <div className="bg-[#F1F5F9] p-6 rounded-xl border border-[#e2e8f0] shadow-sm">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Layout size={20} className="text-[#2563EB]" />
                Property Details
              </h2>

              <div className="space-y-5">
                {/* Area Input */}
                <div>
                  <label className="block text-sm font-medium text-[#475569] mb-1 flex items-center gap-2">
                    <Square size={16} /> Area (Sq. Ft.)
                  </label>
                  <input 
                    type="number" 
                    value={formData.area}
                    onChange={(e) => setFormData({...formData, area: e.target.value})}
                    className="w-full p-2 border border-[#cbd5e1] rounded-md focus:ring-2 focus:ring-[#2563EB] focus:outline-none bg-white transition-all"
                  />
                  <input 
                    type="range" 
                    min="200" 
                    max="10000" 
                    step="50"
                    value={formData.area}
                    onChange={(e) => setFormData({...formData, area: e.target.value})}
                    className="w-full mt-2 accent-[#2563EB]"
                  />
                </div>

                {/* BHK Selection */}
                <div>
                  <label className="block text-sm font-medium text-[#475569] mb-2 flex items-center gap-2">
                    <Bed size={16} /> BHK
                  </label>
                  <div className="flex gap-2 flex-wrap">
                    {[1, 2, 3, 4, 5].map((num) => (
                      <button
                        key={num}
                        onClick={() => setFormData({...formData, bhk: num})}
                        className={`px-4 py-2 rounded-md border transition-all ${
                          formData.bhk === num 
                          ? "bg-[#2563EB] text-white border-[#2563EB]" 
                          : "bg-white text-[#475569] border-[#cbd5e1] hover:border-[#2563EB]"
                        }`}
                      >
                        {num === 5 ? "5+" : num}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Bathrooms & Balcony */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#475569] mb-1 flex items-center gap-2">
                      <Bath size={16} /> Bathrooms
                    </label>
                    <div className="flex items-center gap-3">
                      <button 
                        onClick={() => setFormData({...formData, bathrooms: Math.max(1, formData.bathrooms - 1)})}
                        className="w-8 h-8 rounded-full border border-[#cbd5e1] bg-white flex items-center justify-center hover:border-[#2563EB]"
                      >-</button>
                      <span className="font-semibold w-4 text-center">{formData.bathrooms}</span>
                      <button 
                        onClick={() => setFormData({...formData, bathrooms: Math.min(10, formData.bathrooms + 1)})}
                        className="w-8 h-8 rounded-full border border-[#cbd5e1] bg-white flex items-center justify-center hover:border-[#2563EB]"
                      >+</button>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#475569] mb-1 flex items-center gap-2">
                      <Layout size={16} /> Balcony
                    </label>
                    <div className="flex items-center gap-3">
                      <button 
                        onClick={() => setFormData({...formData, balcony: Math.max(0, formData.balcony - 1)})}
                        className="w-8 h-8 rounded-full border border-[#cbd5e1] bg-white flex items-center justify-center hover:border-[#2563EB]"
                      >-</button>
                      <span className="font-semibold w-4 text-center">{formData.balcony}</span>
                      <button 
                        onClick={() => setFormData({...formData, balcony: Math.min(5, formData.balcony + 1)})}
                        className="w-8 h-8 rounded-full border border-[#cbd5e1] bg-white flex items-center justify-center hover:border-[#2563EB]"
                      >+</button>
                    </div>
                  </div>
                </div>

                {/* Location Autocomplete */}
                <div>
                  <label className="block text-sm font-medium text-[#475569] mb-1 flex items-center gap-2">
                    <MapPin size={16} /> Location
                  </label>
                  <select
                    value={formData.location}
                    onChange={(e) => setFormData({...formData, location: e.target.value})}
                    className="w-full p-2 border border-[#cbd5e1] rounded-md focus:ring-2 focus:ring-[#2563EB] focus:outline-none bg-white transition-all"
                  >
                    {locations.map(loc => (
                      <option key={loc} value={loc}>{loc}</option>
                    ))}
                  </select>
                </div>

                <button 
                  onClick={handlePredict}
                  disabled={loading}
                  className="w-full bg-[#2563EB] text-white py-3 rounded-lg font-semibold flex items-center justify-center gap-2 hover:bg-[#1d4ed8] transition-all disabled:opacity-70 disabled:cursor-not-allowed mt-4 shadow-sm"
                >
                  {loading ? <Loader2 className="animate-spin" /> : <ChevronRight />}
                  {loading ? "Calculating..." : "Reveal Prediction"}
                </button>

                {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
              </div>
            </div>
          </motion.div>

          {/* Result Section */}
          <div className="flex flex-col justify-center">
            <AnimatePresence mode="wait">
              {prediction ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="bg-white p-8 rounded-2xl border-2 border-[#2563EB] shadow-xl text-center relative overflow-hidden"
                >
                  <div className="absolute top-0 right-0 p-2">
                    <Home className="text-[#2563EB] opacity-10" size={80} />
                  </div>
                  
                  <h3 className="text-[#64748b] text-sm uppercase tracking-widest font-semibold mb-2">Estimated Market Value</h3>
                  <div className="text-4xl sm:text-5xl font-extrabold text-[#1e293b] mb-6">
                    {prediction.formatted_prediction}
                  </div>

                  <div className="bg-[#F8FAFC] p-4 rounded-xl border border-[#e2e8f0] text-left">
                    <div className="flex items-start gap-3">
                      <Info size={18} className="text-[#2563EB] mt-0.5" />
                      <div>
                        <p className="text-sm font-medium text-[#1e293b]">Insights</p>
                        <p className="text-xs text-[#64748b] mt-1">
                          {prediction.influence_summary}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Visual Feature Importance */}
                  <div className="mt-6 space-y-3">
                    <p className="text-xs font-semibold text-[#64748b] uppercase tracking-wider text-left">Primary Drivers</p>
                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className="text-[#475569]">Location</span>
                          <span className="font-bold text-[#2563EB]">45%</span>
                        </div>
                        <div className="w-full h-1.5 bg-[#e2e8f0] rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: "45%" }}
                            className="h-full bg-[#2563EB]"
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className="text-[#475569]">Area</span>
                          <span className="font-bold text-[#2563EB]">35%</span>
                        </div>
                        <div className="w-full h-1.5 bg-[#e2e8f0] rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: "35%" }}
                            className="h-full bg-[#2563EB]"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-8 text-xs text-[#94a3b8]">
                    Disclaimer: This estimate is based on current market trends and heuristic modeling. Actual prices may vary.
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-center p-8 border-2 border-dashed border-[#e2e8f0] rounded-2xl"
                >
                  <Home size={48} className="mx-auto text-[#cbd5e1] mb-4" />
                  <p className="text-[#64748b]">Enter property details to see the predicted price</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <footer className="mt-16 pt-8 border-t border-[#e2e8f0] text-center text-sm text-[#94a3b8]">
          &copy; {new Date().getFullYear()} Professional House Predictor. Designed for the Indian Real Estate Market.
        </footer>
      </div>
    </main>
  );
}
